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
from mada.core.database import ChatSessionManager
from mada.core.orchestrator import MADAOrchestrator

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

    async def run(self):
        """Run the interactive CLI session."""
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
                        self.config.agents, self.config.mcp_servers
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
                    print(f"\nWARNING: Initialization error: {e}")
                    print("Continuing with available agents...")
                    import traceback

                    traceback.print_exc()

                print("\nChat with the agents (type 'quit' to exit)")
                print("-" * 50)

                prompt_session = PromptSession()

                # Interactive chat loop
                while True:
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

                        await orchestrator.background_tasks.run_query(
                            user_input, blocking=self.blocking
                        )

                    except KeyboardInterrupt:
                        print("\n\nGoodbye!")
                        break
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        import traceback

                        traceback.print_exc()

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


async def async_main(config_file: str, blocking: bool = False):
    """
    Async main entry point for CLI.

    Args:
        config_file: The path to the MADA configuration file.
        blocking: If True, process one query at a time.
    """
    try:
        # Load configuration
        config = load_config_from_json(config_file)

        # Run CLI
        cli = MADACLIInterface(config, blocking=blocking)
        await cli.run()

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
def main(config_file: str, blocking: bool) -> None:
    """
    Run MADA in CLI mode.

    CONFIG_FILE is the path to the MADA configuration file.
    """
    asyncio.run(async_main(config_file, blocking=blocking))


if __name__ == "__main__":
    main()
