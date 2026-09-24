"""LLM-backed actor and associative learner implementations."""

from __future__ import annotations

import copy
import ast
import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from PIL import Image

from agent_core import token_meter
from agent_core.context import SemanticLog
from agent_core.context_policy import ContextPolicy, validate_context_policy
from agent_core.contracts import extract_json
from agent_core.llm import BaseAgent, LLMConfig, MalformedContentError, build_content
from agent_core.sys_inst import SYS_INST_VLM_LEAN


# Legacy salvage form for actor replies that are not a parseable mapping.
_ACTIONS_RE = re.compile(r"['\"]actions['\"]\s*:\s*\[([^\[\]]*)\]", re.DOTALL)
_TIMES_RE = re.compile(r"['\"]times['\"]\s*:\s*\[([^\[\]]*)\]", re.DOTALL)
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_INT_RE = re.compile(r"-?\d+")


def _client(config: LLMConfig) -> OpenAI:
    return OpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=0)


class AssociativeLearner(BaseAgent):
    """LLM client used to produce semantic decisions and episodic reflections."""
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        super().__init__(config)
        self.client = _client(self.config)

    def generate_content(
        self,
        system_instruction: str,
        image: Optional[Image.Image],
        text: str,
    ) -> str:
        content = build_content(image, text) if image else [{"type": "text", "text": text}]
        return self._api_call_with_retry(
            self.client,
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
        )


SemanticEpisodicAssociativeLearner = AssociativeLearner


class VLMAgent(BaseAgent):
    """Conversational visual actor that retains the active leg's message history."""
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        context_policy: ContextPolicy = ContextPolicy(),
    ) -> None:
        super().__init__(config)
        self.context_policy = validate_context_policy(context_policy)
        self.client = _client(self.config)
        self.history: List[Dict[str, Any]] = []
        # Retained here for compatibility. MemoryRuntime owns mutation/persistence at runtime.
        self.episodic_memory = ""
        self.semantic_log = SemanticLog("", self.context_policy)
        logger.info(f"VLMAgent initialized with model: {self.config.model_id}")

    def reset_history(self) -> None:
        self.history = []

    def send_message(self, content: list) -> str:
        self.history.append({"role": "user", "content": content})
        try:
            return self._complete_turn()
        except BaseException:
            self.history.pop()  # keep user/assistant turns alternating after a failed call
            raise

    def _complete_turn(self) -> str:
        messages = [
            {"role": "system", "content": SYS_INST_VLM_LEAN},
            *self._outbound_history(),
        ]
        def validate(raw: str) -> str:
            blob = extract_json(self.extractable_json_structured_output, str(raw or ""))
            parsed = None
            for parser in (ast.literal_eval, json.loads):
                try:
                    candidate = parser(blob)
                except Exception:  # the actor contract also permits the legacy salvage form
                    continue
                if isinstance(candidate, dict) and "actions" in candidate:
                    parsed = candidate
                    break
            if parsed is None:
                actions = _ACTIONS_RE.search(blob)
                times = _TIMES_RE.search(blob)
                action_items = _QUOTED_RE.findall(actions.group(1)) if actions else []
                time_items = _INT_RE.findall(times.group(1)) if times else []
                if not action_items or len(action_items) != len(time_items):
                    raise MalformedContentError(
                        "actor response did not contain usable actions and times", content=raw
                    )
            return raw

        try:
            with token_meter.role(token_meter.ROLE_ACTOR):
                completion = self._api_call_result_with_retry(
                    self.client, messages, call_name="actor_reasoning", validator=validate
                )
                reply = completion.text or ""
                assistant_message = completion.assistant_message
        except MalformedContentError as error:
            # Preserve the existing actor-step error path after the shared budget is exhausted.
            reply = str(error.content or "")
            completion = error.completion_result
            assistant_message = (
                completion.assistant_message if completion is not None
                else {"role": "assistant", "content": reply}
            )
        # A None-content reply drops the key; history readers and resends need it.
        self.history.append({**assistant_message, "content": assistant_message.get("content") or ""})
        return reply

    def _outbound_history(self) -> list[dict[str, Any]]:
        keep = self.context_policy.actor_image_history
        if keep is None:
            return self.history

        user_indices = [
            index for index, message in enumerate(self.history) if message.get("role") == "user"
        ]
        newest = set(user_indices[-keep:])
        outbound: list[dict[str, Any]] = []
        for index, message in enumerate(self.history):
            content = message.get("content")
            if index not in newest and message.get("role") == "user" and isinstance(content, list):
                filtered = [
                    copy.deepcopy(part)
                    for part in content
                    if not (isinstance(part, dict) and part.get("type") == "image_url")
                ]
                outbound.append({**message, "content": filtered})
            else:
                outbound.append(message)
        return outbound

    def get_history_text(self, n: int = 8) -> str:
        result = ""
        for message in self.history[-n:]:
            role = message["role"]
            content = message.get("content") or ""
            if isinstance(content, list):
                text = " ".join(part["text"] for part in content if part.get("type") == "text")
            else:
                text = content
            result += f"{role.upper()}: {text}\n"
        return result.strip()
