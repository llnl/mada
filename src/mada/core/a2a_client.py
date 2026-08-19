# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Small A2A client used by the MADA orchestrator.

This is the client-side A2A helper: the orchestrator uses it to call remote
A2A agents configured under `a2a.agents`. The server-side interface that
exposes MADA itself as an A2A agent lives in `mada.interfaces.a2a.main` and
uses the `a2a.self` configuration block.
"""

from __future__ import annotations

from typing import Any

import httpx
from a2a.client import A2ACardResolver
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.types import AgentCard
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from agent_framework.a2a import A2AAgent
from google.protobuf.json_format import ParseDict

from mada.core.config import RemoteA2AAgentConfig


class RemoteA2AClient:
    """
    Minimal client for delegating a text task to a remote A2A agent.
    """

    def __init__(self, name: str, config: RemoteA2AAgentConfig) -> None:
        """
        Initialize an HTTP client for a configured remote A2A agent.
        """
        self.name = name
        self.config = config
        self._headers = dict(config.headers)
        if config.api_key:
            self._headers["x-api-key"] = config.api_key
        self._client = httpx.AsyncClient(headers=self._headers, timeout=config.timeout)
        self._agent_card: AgentCard | None = None

    async def send_message(self, task: str) -> str:
        """
        Send a text task to the remote A2A agent and return its text response.
        """
        if self._agent_card is None:
            await self.get_agent_card()
        agent_card = self._agent_card
        if agent_card is None:
            raise RuntimeError(f"A2A agent card was not loaded for {self.name}")
        agent = A2AAgent(
            name=agent_card.name or self.name,
            url=self.config.url,
            agent_card=agent_card,
            http_client=self._client,
        )
        response = await agent.run(task)
        return self._extract_text(response)

    async def get_agent_card(self) -> dict[str, Any]:
        """
        Fetch the remote agent card when the A2A server exposes one.
        """
        if self._agent_card is None:
            if self.config.card_url:
                response = await self._client.get(self.config.card_url, timeout=5.0)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(
                        "A2A agent card response must be a JSON object: "
                        f"{self.config.card_url}"
                    )
                self._agent_card = ParseDict(data, AgentCard())
            else:
                base_url = self.config.url.rstrip("/")
                if base_url.endswith("/a2a"):
                    base_url = base_url[: -len("/a2a")]
                resolver = A2ACardResolver(
                    httpx_client=self._client,
                    base_url=base_url,
                )
                self._agent_card = await resolver.get_agent_card()

            if not any(
                interface.protocol_binding == TransportProtocol.JSONRPC.value
                and interface.protocol_version == PROTOCOL_VERSION_1_0
                for interface in self._agent_card.supported_interfaces
            ):
                raise RuntimeError(
                    f"A2A agent {self.name} must advertise a JSONRPC "
                    f"{PROTOCOL_VERSION_1_0} supported interface."
                )

        return self._to_dict(self._agent_card)

    async def aclose(self) -> None:
        """
        Close the underlying async HTTP client.
        """
        await self._client.aclose()

    def _extract_text(self, result: Any) -> str:
        """
        Extract human-readable text from an A2A response payload.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result

        texts = []
        self._collect_agent_framework_text(result, texts)
        if texts:
            return "\n".join(texts)
        self._collect_text_parts(result, texts)
        if texts:
            return "\n".join(texts)
        return str(result)

    def _collect_agent_framework_text(self, value: Any, texts: list[str]) -> None:
        """
        Collect common Agent Framework text fields from response objects.
        """
        for attr in ("messages", "contents"):
            items = getattr(value, attr, None)
            if isinstance(items, list):
                for item in items:
                    self._collect_agent_framework_text(item, texts)

        text = getattr(value, "text", None)
        if text:
            texts.append(str(text))

    def _collect_text_parts(self, value: Any, texts: list[str]) -> None:
        """
        Recursively collect text parts from an A2A result structure.
        """
        if not isinstance(value, (dict, list)):
            value = self._to_dict(value)
            if not value:
                return

        if isinstance(value, dict):
            parts = value.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("kind") == "text" or part.get("type") == "text":
                        text = part.get("text")
                        if text:
                            texts.append(str(text))
            for item in value.values():
                self._collect_text_parts(item, texts)
        elif isinstance(value, list):
            for item in value:
                self._collect_text_parts(item, texts)

    def _to_dict(self, value: Any) -> dict[str, Any]:
        """
        Convert A2A SDK and Pydantic objects to plain dictionaries.
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, AgentCard):
            return agent_card_to_dict(value)
        data = value.dict(by_alias=True)
        if not isinstance(data, dict):
            raise RuntimeError("A2A SDK object did not serialize to a dictionary")
        return data
