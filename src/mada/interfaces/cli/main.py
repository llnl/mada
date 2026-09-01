# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple CLI interface using the core orchestrator.

This provides a clean command-line interface that uses the extracted
core orchestration logic rather than duplicating it.
"""

import asyncio
import sys

import click
from typing import Dict, List

from mada.core.config import AppConfig, OrchestrationConfig, load_config_from_json
from mada.core.autonomy import (
    MAX_AUTONOMY_WAIT_SECONDS,
    build_autonomy_enabled_prompt,
    build_autonomy_followup_prompt,
    default_wait_seconds_from_user_message,
    max_autonomy_followups,
    parse_autonomy_control,
    tail_text,
)
from mada.core.database import ChatSessionManager
from mada.core.orchestrator import MADAOrchestrator
from mada.core.orchestration.stream_events import apply_text_control

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout


class MADACLIInterface:
    """
    Simple command-line interface for MADA.

    Uses the core MADAOrchestrator for all orchestration logic,
    providing a clean separation between UI and core functionality.
    """

    def __init__(self, config: AppConfig, blocking: bool = False):
        """
        Initialize the CLI with configuration.

        Args:
            config: Application configuration
            blocking: If True, wait for each response inline. If False,
                submit queries in the background and return to the prompt.
        """
        self.config = config
        self.blocking = blocking
        self.orchestrator = None
        self.session_manager = ChatSessionManager(config.database)

    @property
    def orchestration_config(self) -> OrchestrationConfig:
        return getattr(self.config, "orchestration", None) or OrchestrationConfig()

    def _print_history_summary(self, history: List[Dict[str, str]]):
        """
        Print a brief summary of session history for context.
        Assumes `history` is a list of dicts like:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Args:
            history: A list of the chat history
        """
        if not history:
            print("This session has no previous messages.")
            return

        print("\nPrevious messages in this session (most recent last):")
        # Show at most the last 5 turns
        max_messages = 10
        to_show = history[-max_messages:]
        for msg in to_show:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            # Truncate very long lines for CLI readability
            if len(content) > 200:
                content = content[:197] + "..."
            print(f"  {role}: {content}")

    def startup_session_menu(self) -> bool:
        """
        Interactive menu on startup to manage chat sessions.

        Returns:
            bool: True if a session has been selected/created and we should continue,
                False if the user chose to quit.
        """
        while True:
            print("\nChat Session Manager")
            print("=" * 50)
            sessions = self.list_sessions()

            if sessions:
                print("Existing sessions:")
                for idx, label in enumerate(sessions, start=1):
                    print(f"  {idx}. {label}")
            else:
                print("No existing sessions.")

            print("\nOptions:")
            print("  [n] New session")
            if sessions:
                print("  [s] Select session")
                print("  [d] Delete session")
                print("  [a] Delete ALL sessions")
            print("  [q] Quit")

            choice = input("\nEnter choice: ").strip().lower()

            if choice == "q":
                return False

            if choice == "n":
                self.create_new_session()
                print("Created and selected a new session.")
                return True

            if choice == "s" and sessions:
                try:
                    selection = input(
                        "Enter the number of the session to select: "
                    ).strip()
                    idx = int(selection)
                    if 1 <= idx <= len(sessions):
                        label = sessions[idx - 1]
                        history = self.select_session(label)
                        print(f"Selected session: {label}")
                        self._print_history_summary(history)
                        return True
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Please enter a valid number.")
                continue

            if choice == "d" and sessions:
                try:
                    selection = input(
                        "Enter the number of the session to delete: "
                    ).strip()
                    idx = int(selection)
                    if 1 <= idx <= len(sessions):
                        label = sessions[idx - 1]
                        confirm = (
                            input(f"Are you sure you want to delete '{label}'? [y/N]: ")
                            .strip()
                            .lower()
                        )
                        if confirm == "y":
                            self.delete_session(label)
                            print("Session deleted.")
                        else:
                            print("Deletion cancelled.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Please enter a valid number.")
                continue

            if choice == "a" and sessions:
                print(
                    f"\nWARNING: This will delete ALL {len(sessions)} session(s) and cannot be undone!"
                )
                confirm = (
                    input(
                        "Are you absolutely sure you want to delete ALL sessions? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if confirm == "y":
                    # Use confirm=False since we already confirmed
                    self.session_manager.delete_all_sessions(confirm=False)
                    print("All sessions deleted.")
                else:
                    print("Deletion cancelled.")
                continue

            print("Invalid choice, please try again.")

    def list_sessions(self) -> List[str]:
        """
        List all sessions available and create a list of labels for each session.

        Returns:
            A list of labels of the form "timestamp | ID" for each session.
        """
        sessions = self.session_manager.list_sessions()
        return [
            f"{ts.isoformat(sep=' ', timespec='seconds')} | {sid}"
            for sid, ts in sessions
        ]

    def create_new_session(self):
        """
        Create a new chat session.
        """
        new_id = self.session_manager.create_session_id()
        self.session_manager.create_new_session(new_id)
        self.session_manager.select_session(new_id)

    def _extract_id_from_label(self, session_label: str) -> str:
        """
        Extract session id from label.

        Args:
            session_label (str): The label of the session of the form "timestamp | ID"

        Returns:
            The session ID.
        """
        try:
            session_id = session_label.split("|", 1)[1].strip()
        except Exception:
            session_id = session_label.strip()

        return session_id

    def select_session(self, session_label: str) -> List[Dict[str, str]]:
        """
        Given a label like 'timestamp | session_id', select that session and return its history.

        Args:
            session_label (str): The label of the chat session that the user wants to select.

        Returns:
            A list of the chat history.
        """
        if not session_label:
            return []  # nothing selected

        # Extract session id from label
        session_id = self._extract_id_from_label(session_label)

        history = self.session_manager.select_session(session_id)

        return history

    def delete_session(self, session_label: str):
        """
        Delete a chat session.

        Args:
            session_label (str): The label of the chat session that the user wants to delete.
        """
        session_id = self._extract_id_from_label(session_label)
        self.session_manager.delete_session(session_id)

    async def run(
        self,
        autonomy_level: int = 0,
        show_autonomy_debug: bool = False,
    ):
        """Run the interactive CLI session."""
        if autonomy_level > 0 and not self.blocking:
            raise ValueError(
                "--autonomy-level requires --blocking because autonomous follow-ups "
                "cannot be detached safely"
            )

        print("MADA Multi-Agent Orchestrator")
        print("=" * 50)

        # Startup session menu
        should_continue = self.startup_session_menu()
        if not should_continue:
            print("No session selected. Exiting.")
            return

        # Create orchestrator
        try:
            async with MADAOrchestrator(
                model_config=self.config.model,
                database_config=self.config.database,
                session_manager=self.session_manager,
                orchestration_config=self.orchestration_config,
            ) as orchestrator:
                self.orchestrator = orchestrator

                # Setup agents - continue even if some MCP servers fail
                print("\nInitializing agents and MCP servers...")
                try:
                    status, tools = await orchestrator.initialize_orchestrator(
                        self.config.agents,
                        self.config.mcp_servers,
                        getattr(self.config, "a2a_agents", {}),
                    )
                    print(f"Status: {status}")
                    print(f"Orchestration mode: {self.orchestration_config.mode}")
                    print(
                        f"Model: {self.config.model.model} from {self.config.model.provider}"
                    )

                    if tools:
                        print("\nAvailable tools:")
                        for tool in tools[:10]:  # Show first 10 tools
                            print(f"   • {tool}")
                        if len(tools) > 10:
                            print(f"   ... and {len(tools) - 10} more tools")
                    else:
                        print("\nNo MCP tools available (LLM-only agents)")

                except BaseExceptionGroup as eg:
                    print(
                        f"\nWARNING: {len(eg.exceptions)} initialization error(s) occurred. Continuing with available agents..."
                    )
                except Exception as e:
                    print(f"\nERROR: Initialization failed: {e}")
                    return

                print("\nChat with the agents (type 'quit' to exit)")
                print("-" * 50)

                prompt_session = PromptSession()
                pending_clarification: str | None = None

                # Interactive chat loop
                while True:
                    # Initialize turn-local state before any operation that
                    # can fail so the error path cannot reuse a prior turn.
                    user_input = ""
                    level = max(0, min(9, int(autonomy_level or 0)))
                    assistant_buffer = ""
                    assistant_reply = ""
                    session_id = self.session_manager.current_session_id
                    try:
                        with patch_stdout():
                            user_input = (
                                await prompt_session.prompt_async("\nYou: ")
                            ).strip()

                        if user_input.lower() in ["quit", "exit", "q"]:
                            pending_count = await orchestrator.background_tasks.count_pending_tasks()
                            if pending_count:
                                print(
                                    f"\nExiting with {pending_count} pending background task(s)."
                                )
                            print("\nGoodbye!")
                            break

                        if not user_input:
                            continue

                        if not self.blocking and user_input.lower() == "tasks":
                            task_snapshot = (
                                await orchestrator.background_tasks.get_task_snapshot()
                            )
                            pending = []
                            finished = []
                            for task_id, task in task_snapshot.items():
                                status = task.get("status", "unknown")
                                tool_name = task.get("tool_name", "unknown tool")
                                task_type = task.get("type", "task")
                                line = f"{task_id}: {status}, {task_type}, tool={tool_name}"
                                if status in ("pending", "running"):
                                    pending.append(line)
                                else:
                                    finished.append(line)

                            print("\nPending tasks:")
                            for line in pending:
                                print(f"  - {line}")
                            if not pending:
                                print("  none")

                            print("Finished tasks:")
                            for line in finished:
                                print(f"  - {line}")
                            if not finished:
                                print("  none")
                            continue

                        print("\nAgents:")
                        print("-" * 20)

                        model_user_input = user_input
                        had_pending_clarification = pending_clarification is not None
                        turn_pending_clarification = pending_clarification
                        if pending_clarification:
                            clarification_context = (
                                "Previous question you asked the user:\n"
                                f"{pending_clarification}\n\n"
                                "User answer:\n"
                                f"{user_input}"
                            )
                        else:
                            clarification_context = model_user_input

                        if level <= 0:
                            query_kwargs = {"blocking": self.blocking}
                            if had_pending_clarification:
                                query_kwargs["persistence_user_input"] = user_input
                            await orchestrator.background_tasks.run_query(
                                clarification_context, **query_kwargs
                            )
                            if pending_clarification == turn_pending_clarification:
                                pending_clarification = None
                            continue

                        max_followups = max_autonomy_followups(level)
                        assistant_buffer = ""
                        model_reply_buffer = ""
                        last_reply = ""
                        # Autonomy prompts are internal scaffolding; keep them
                        # out of the shared provider session. The visible turn
                        # is persisted by this interface after the loop.
                        is_magentic = self.orchestration_config.mode == "magentic"
                        isolated_session = True
                        autonomy_prompt = build_autonomy_enabled_prompt(
                            clarification_context,
                            level=level,
                            followups_used=0,
                            followups_max=max_followups,
                        )

                        try:
                            response_chunks: list[str] = []
                            stream_terminal = False
                            response_prefix = assistant_buffer
                            session_id = self.session_manager.current_session_id
                            async for chunk in orchestrator.process_message(
                                autonomy_prompt,
                                isolated_session=isolated_session,
                                persistence_session_id=session_id,
                                record_to_db=False,
                                background_poll_session_id=session_id,
                            ):
                                handled, terminal = apply_text_control(
                                    response_chunks, chunk
                                )
                                if handled:
                                    last_reply = "".join(response_chunks)
                                    model_reply_buffer = last_reply
                                    assistant_buffer = response_prefix + last_reply
                                    # Print replacement/error text so terminal shows corrected output.
                                    # In non-magentic mode, provisional chunks were already printed;
                                    # we append the authoritative text so users see the final answer.
                                    if not is_magentic:
                                        print(last_reply, end="", flush=True)
                                    if terminal:
                                        stream_terminal = True
                                        break
                                    continue
                                content = str(chunk)
                                response_chunks.append(content)
                                if not is_magentic:
                                    last_reply += content
                                    model_reply_buffer += content
                                    assistant_buffer += content
                                    print(content, end="", flush=True)
                            if is_magentic:
                                last_reply = "".join(response_chunks)
                                model_reply_buffer = last_reply
                                assistant_buffer += last_reply
                                print(last_reply, end="", flush=True)
                            if stream_terminal:
                                max_followups = 0
                        except asyncio.CancelledError:
                            marker = "\n\n---\n\n[Generation cancelled]\n"
                            print(marker, end="", flush=True)
                            assistant_buffer += marker
                            self.session_manager.add_message("user", user_input)
                            self.session_manager.add_message(
                                "assistant",
                                model_reply_buffer or "[Generation cancelled]",
                            )
                            orchestrator.clear_deferred_background_polls(session_id)
                            print("\n")
                            continue

                        followup_count = 0
                        while followup_count < max_followups:
                            decision_prompt = (
                                "[INTERNAL CONTROL MESSAGE]\n"
                                "Choose exactly one:\n"
                                "- WAIT: sleep briefly and then issue a follow-up query\n"
                                "- CONTINUE: immediately issue a follow-up query\n"
                                "- ASK: ask the user ONE question (only if truly blocked)\n"
                                "- STOP: stop\n\n"
                                f"Autonomy level: {level} (0=off, 9=most autonomous)\n"
                                f"Follow-up safety cap used: {followup_count}/{max_followups}\n\n"
                                "User request:\n"
                                f"{model_user_input}\n\n"
                                "Assistant reply:\n"
                                f"{(last_reply or '').strip()}\n\n"
                                "Assistant transcript so far (tail):\n"
                                f"{tail_text(assistant_buffer, 2000).strip()}\n\n"
                                "Important:\n"
                                "- The system CAN wait in real time. Do NOT claim you cannot wait.\n"
                                "- If the user requested timed repetition (e.g., 'every 20 seconds'), choose WAIT and set AUTONOMY_WAIT_SECONDS accordingly.\n"
                                "- If the user requested 'until sufficient', choose a reasonable stopping point and STOP once you reach it.\n"
                                "- Do not include explanations, code fences, or any other text.\n\n"
                                "Output EXACTLY this format, with no extra text:\n"
                                "AUTONOMY_DECISION=<CONTINUE|WAIT|ASK|STOP>\n"
                                "AUTONOMY_QUERY=<single-line query to run next>\n"
                                "AUTONOMY_WAIT_SECONDS=<integer seconds to wait (WAIT only)>\n"
                                "AUTONOMY_QUESTION=<single-line question for the user>\n"
                                "Rules:\n"
                                "- If decision is CONTINUE, set AUTONOMY_QUERY and leave AUTONOMY_QUESTION empty.\n"
                                "- If decision is WAIT, set AUTONOMY_WAIT_SECONDS and AUTONOMY_QUERY, and leave AUTONOMY_QUESTION empty.\n"
                                "- If decision is ASK, set AUTONOMY_QUESTION and leave AUTONOMY_QUERY empty.\n"
                                "- If decision is STOP, leave all other fields empty.\n"
                            )

                            decision_text = await orchestrator.run_control_prompt(
                                decision_prompt
                            )
                            (
                                decision,
                                next_query,
                                question,
                                wait_seconds,
                                parse_ok,
                            ) = parse_autonomy_control(decision_text)

                            if show_autonomy_debug:
                                debug_text = (
                                    "\n\n---\n\n"
                                    f"[Autonomy debug] level={level} parse_ok={parse_ok} "
                                    f"decision={decision} next_query={'yes' if bool(next_query) else 'no'} "
                                    f"wait_seconds={wait_seconds or 0} question={'yes' if bool(question) else 'no'}\n"
                                )
                                print(debug_text, end="", flush=True)
                                assistant_buffer += debug_text

                            if decision == "ASK" and question:
                                question_text = f"\n\n{question}\n"
                                print(question_text, end="", flush=True)
                                assistant_buffer += question_text
                                model_reply_buffer = (
                                    f"{model_reply_buffer}\n\n{question}".strip()
                                )
                                pending_clarification = question
                                break

                            if decision == "STOP":
                                break

                            if decision not in ("CONTINUE", "WAIT") or not next_query:
                                break

                            if decision == "WAIT":
                                if wait_seconds <= 0:
                                    wait_seconds = (
                                        default_wait_seconds_from_user_message(
                                            user_input
                                        )
                                    )
                                if wait_seconds > MAX_AUTONOMY_WAIT_SECONDS:
                                    wait_seconds = MAX_AUTONOMY_WAIT_SECONDS

                                marker = (
                                    f"\n\n---\n\n[Autonomous wait] {wait_seconds}s\n"
                                )
                                print(marker, end="", flush=True)
                                assistant_buffer += marker

                                try:
                                    await asyncio.sleep(wait_seconds)
                                except asyncio.CancelledError:
                                    interrupted = "\n\n[Autonomous wait interrupted]\n"
                                    print(interrupted, end="", flush=True)
                                    assistant_buffer += interrupted
                                    break

                            marker = f"\n\n---\n\n[Autonomous query] {next_query}\n\n"
                            print(marker, end="", flush=True)
                            assistant_buffer += marker

                            previous_reply = last_reply
                            last_reply = ""
                            followup_prompt = build_autonomy_followup_prompt(
                                next_query,
                                original_request=model_user_input,
                                last_reply=previous_reply,
                                assistant_buffer=assistant_buffer,
                                level=level,
                                followups_used=followup_count,
                                followups_max=max_followups,
                            )
                            try:
                                response_chunks = []
                                followup_reply = ""
                                stream_terminal = False
                                response_prefix = assistant_buffer
                                async for chunk in orchestrator.process_message(
                                    followup_prompt,
                                    isolated_session=isolated_session,
                                    persistence_session_id=session_id,
                                    record_to_db=False,
                                    background_poll_session_id=session_id,
                                ):
                                    handled, terminal = apply_text_control(
                                        response_chunks, chunk
                                    )
                                    if handled:
                                        last_reply = "".join(response_chunks)
                                        followup_reply = last_reply
                                        assistant_buffer = response_prefix + last_reply
                                        if not is_magentic:
                                            print(last_reply, end="", flush=True)
                                        if terminal:
                                            stream_terminal = True
                                            break
                                        continue
                                    content = str(chunk)
                                    response_chunks.append(content)
                                    if not is_magentic:
                                        last_reply += content
                                        followup_reply += content
                                        assistant_buffer += content
                                        print(content, end="", flush=True)
                                if is_magentic:
                                    last_reply = "".join(response_chunks)
                                    followup_reply = last_reply
                                    assistant_buffer += last_reply
                                    print(last_reply, end="", flush=True)
                                model_reply_buffer += followup_reply
                                if stream_terminal:
                                    followup_count = max_followups
                            except asyncio.CancelledError:
                                marker = "\n\n---\n\n[Generation cancelled]\n"
                                print(marker, end="", flush=True)
                                assistant_buffer += marker
                                break

                            followup_count += 1

                        # Persist every autonomy turn, including empty model
                        # completions, so the user's request remains in history.
                        self.session_manager.add_message("user", user_input)
                        self.session_manager.add_message(
                            "assistant",
                            model_reply_buffer or "[No response]",
                        )
                        orchestrator.start_deferred_background_polls(
                            session_id,
                            model_reply_buffer or "[No response]",
                        )
                        if pending_clarification == turn_pending_clarification:
                            pending_clarification = None
                        print("\n")

                    except KeyboardInterrupt:
                        if level > 0 and session_id:
                            orchestrator.clear_deferred_background_polls(session_id)
                        print("\n\nGoodbye!")
                        break
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        import traceback

                        traceback.print_exc()
                        if level > 0 and user_input and session_id:
                            error_msg = f"Error processing message: {e}"
                            assistant_reply = (
                                f"{model_reply_buffer}\n\n{error_msg}"
                                if model_reply_buffer.strip()
                                else error_msg
                            )
                            self.session_manager.add_message("user", user_input)
                            self.session_manager.add_message(
                                "assistant", assistant_reply
                            )
                            orchestrator.clear_deferred_background_polls(session_id)

        except BaseExceptionGroup as eg:
            print(
                f"\nFATAL ERROR: Multiple errors creating orchestrator ({len(eg.exceptions)} errors)"
            )
            import traceback

            traceback.print_exc()
            return
        except Exception as e:
            print(f"\nFATAL ERROR: Failed to create orchestrator: {e}")
            import traceback

            traceback.print_exc()
            return

        # Note: Cleanup is handled automatically by the 'async with' context manager


async def async_main(
    config_file: str,
    blocking: bool = False,
    autonomy_level: int = 0,
    show_autonomy_debug: bool = False,
):
    """
    Async main entry point for CLI.

    Args:
        config_file: The path to the MADA configuration file.
        blocking: If True, process one query at a time.
        autonomy_level: 0 disables autonomous follow-ups; 1-9 enables bounded
            autonomous follow-up turns.
        show_autonomy_debug: If True, print autonomy control details inline.
    """
    try:
        # Load configuration
        config = load_config_from_json(config_file)

        # Run CLI
        cli = MADACLIInterface(config, blocking=blocking)
        await cli.run(
            autonomy_level=autonomy_level,
            show_autonomy_debug=show_autonomy_debug,
        )

    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


@click.command(
    name="mada-cli",
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)
@click.argument(
    "config_file",
    type=str,
)
@click.option(
    "--blocking",
    is_flag=True,
    help="Process one query at a time instead of running queries in the background.",
)
@click.option(
    "--autonomy-level",
    type=int,
    default=0,
    show_default=True,
    help="0 disables autonomy; 1-9 enables bounded autonomous follow-ups.",
)
@click.option(
    "--show-autonomy-debug",
    is_flag=True,
    default=False,
    help="Print autonomy control parsing and decision details.",
)
def main(
    config_file: str,
    blocking: bool,
    autonomy_level: int,
    show_autonomy_debug: bool,
) -> None:
    """
    Run MADA in CLI mode.

    CONFIG_FILE is the path to the MADA configuration file.
    """
    asyncio.run(
        async_main(
            config_file,
            blocking=blocking,
            autonomy_level=autonomy_level,
            show_autonomy_debug=show_autonomy_debug,
        )
    )


if __name__ == "__main__":
    main()
