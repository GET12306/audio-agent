import json
import os
from typing import Any, Dict, List, Optional, Sequence

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

_OPENAI_PARAMS = {"tools", "tool_choice", "stop", "response_format", "max_tokens", "temperature"}


class DeepSeekReasoning(BaseChatModel):
    """LangChain chat model for DeepSeek, preserving reasoning_content."""

    model: str = "deepseek-v4-flash"
    temperature: float = 0
    max_retries: int = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        object.__setattr__(self, "_openai_client", client)

    @property
    def _llm_type(self) -> str:
        return "deepseek-reasoning"

    def bind_tools(
        self,
        tools: Sequence,
        *,
        tool_choice: Optional[dict] = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted_tools = []
        for t in tools:
            if isinstance(t, BaseTool):
                schema = {}
                try:
                    schema = t.args_schema.schema() if t.args_schema else {}
                except Exception:
                    schema = {"type": "object", "properties": {}}
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": schema,
                    },
                })
            else:
                formatted_tools.append(t)
        kwargs["tools"] = formatted_tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(**kwargs)

    def _convert_message(self, msg: BaseMessage) -> Dict[str, Any]:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        if isinstance(msg, ToolMessage):
            return {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
        if isinstance(msg, AIMessage):
            d: Dict[str, Any] = {"role": "assistant", "content": msg.content}
            rc = msg.additional_kwargs.get("reasoning_content")
            if rc:
                d["reasoning_content"] = rc
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)},
                    }
                    for tc in msg.tool_calls
                ]
            return d
        return {"role": "user", "content": str(msg.content)}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if stop:
            body["stop"] = stop
        for k in _OPENAI_PARAMS:
            if k in kwargs:
                body[k] = kwargs[k]

        response = self._openai_client.chat.completions.create(**body)
        choice = response.choices[0]

        ai_msg = AIMessage(content=choice.message.content or "")

        rc = getattr(choice.message, "reasoning_content", None)
        if rc:
            ai_msg.additional_kwargs["reasoning_content"] = rc

        if choice.message.tool_calls:
            ai_msg.tool_calls = [
                {
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "id": tc.id,
                }
                for tc in choice.message.tool_calls
            ]

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
