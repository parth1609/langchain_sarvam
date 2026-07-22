"""Test Sarvam Chat API wrapper."""

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langchain_core.load as lc_load
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_sarvam.chat_models import (
    ChatSarvam,
    _SARVAM_SDK_PARAMS,
    _convert_dict_to_message,
)

if "SARVAM_API_KEY" not in os.environ:
    os.environ["SARVAM_API_KEY"] = "fake-key"


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_sarvam_model_param() -> None:
    llm = ChatSarvam(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = ChatSarvam(model_name="bar")  # type: ignore[call-arg]
    assert llm.model_name == "bar"


def _mock_completion() -> dict:
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar Baz",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def test_sarvam_invoke() -> None:
    llm = ChatSarvam(model="foo")
    mock_client = MagicMock()
    completed = False

    def mock_completions(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return _mock_completion()

    mock_client.completions = mock_completions
    with patch.object(llm, "client", mock_client):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


try:
    import pytest_asyncio as _pytest_asyncio  # noqa: F401
    _has_asyncio_plugin = True
except ImportError:
    _has_asyncio_plugin = False


@pytest.mark.skipif(not _has_asyncio_plugin, reason="pytest-asyncio not installed")
@pytest.mark.asyncio
@pytest.mark.enable_socket
async def test_sarvam_ainvoke() -> None:
    llm = ChatSarvam(model="foo")
    mock_client = AsyncMock()
    completed = False

    async def mock_completions(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return _mock_completion()

    mock_client.completions = mock_completions
    with patch.object(llm, "async_client", mock_client):
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


def test_chat_sarvam_invalid_streaming_params() -> None:
    with pytest.raises(ValueError):
        ChatSarvam(model="foo", streaming=True, n=2)


def test_chat_sarvam_secret() -> None:
    secret = "secretKey"  # noqa: S105
    not_secret = "safe"  # noqa: S105
    llm = ChatSarvam(model="foo", api_key=secret, model_kwargs={"not_secret": not_secret})  # type: ignore[call-arg, arg-type]
    stringified = str(llm)
    assert not_secret in stringified
    assert secret not in stringified


def test_sarvam_serialization() -> None:
    api_key1 = "top secret"
    api_key2 = "topest secret"
    llm = ChatSarvam(model="foo", api_key=api_key1, temperature=0.5)  # type: ignore[call-arg, arg-type]
    dump = lc_load.dumps(llm)
    llm2 = lc_load.loads(
        dump,
        allowed_objects=[ChatSarvam],
        valid_namespaces=["langchain_sarvam"],
        secrets_map={"SARVAM_API_KEY": api_key2},
    )

    assert type(llm2) is ChatSarvam

    assert llm.sarvam_api_key is not None
    assert llm.sarvam_api_key.get_secret_value() not in dump
    assert llm2.sarvam_api_key is not None
    assert llm2.sarvam_api_key.get_secret_value() == api_key2

    assert llm.temperature == llm2.temperature


# ---------------------------------------------------------------------------
# NEW: _SARVAM_SDK_PARAMS allowlist filtering
# ---------------------------------------------------------------------------

def test_sdk_params_allowlist_contains_expected_keys() -> None:
    """_SARVAM_SDK_PARAMS must include all documented SDK parameters."""
    required = {
        "model", "temperature", "top_p", "reasoning_effort",
        "max_tokens", "stream", "stop", "n", "seed",
        "frequency_penalty", "presence_penalty", "wiki_grounding",
        "tools", "tool_choice", "request_options",
    }
    assert required.issubset(_SARVAM_SDK_PARAMS)


def test_unsupported_kwargs_are_filtered_before_sdk_call() -> None:
    """response_format and ls_structured_output_format must NOT reach the SDK."""
    llm = ChatSarvam(model="foo")
    mock_client = MagicMock()
    captured_kwargs: dict = {}

    def mock_completions(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return _mock_completion()

    mock_client.completions = mock_completions

    with patch.object(llm, "client", mock_client):
        llm.invoke(
            "bar",
            response_format={"type": "json_object"},
            ls_structured_output_format={"kwargs": {}, "schema": {}},
        )

    assert "response_format" not in captured_kwargs
    assert "ls_structured_output_format" not in captured_kwargs


def test_known_sdk_kwargs_are_passed_through() -> None:
    """Known params like temperature and seed must still reach the SDK."""
    llm = ChatSarvam(model="foo", temperature=0.5)
    mock_client = MagicMock()
    captured_kwargs: dict = {}

    def mock_completions(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return _mock_completion()

    mock_client.completions = mock_completions

    with patch.object(llm, "client", mock_client):
        llm.invoke("bar", seed=42)

    assert captured_kwargs.get("temperature") == 0.5
    assert captured_kwargs.get("seed") == 42


# ---------------------------------------------------------------------------
# NEW: _convert_dict_to_message — tool_calls extraction
# ---------------------------------------------------------------------------

def test_convert_dict_to_message_no_tool_calls() -> None:
    """Assistant message with no tool_calls produces AIMessage with empty tool_calls."""
    msg = _convert_dict_to_message({"role": "assistant", "content": "Hello"})
    assert isinstance(msg, AIMessage)
    assert msg.content == "Hello"
    assert msg.tool_calls == []
    assert msg.additional_kwargs.get("tool_calls") is None


def test_convert_dict_to_message_with_dict_tool_calls() -> None:
    """Dict-style tool_calls (as returned by model_dump()) are parsed correctly."""
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Mumbai"}',
                },
            }
        ],
    }
    msg = _convert_dict_to_message(raw)

    assert isinstance(msg, AIMessage)
    assert msg.content == ""
    assert len(msg.tool_calls) == 1

    tc = msg.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Mumbai"}
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "tool_call"

    # additional_kwargs should mirror the raw tool_calls for compatibility
    assert "tool_calls" in msg.additional_kwargs


def test_convert_dict_to_message_multiple_tool_calls() -> None:
    """Multiple tool_calls in one response are all parsed."""
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool_a", "arguments": '{"x": 1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool_b", "arguments": '{"y": "hello"}'},
            },
        ],
    }
    msg = _convert_dict_to_message(raw)

    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0]["name"] == "tool_a"
    assert msg.tool_calls[0]["args"] == {"x": 1}
    assert msg.tool_calls[1]["name"] == "tool_b"
    assert msg.tool_calls[1]["args"] == {"y": "hello"}


def test_convert_dict_to_message_tool_calls_invalid_json_args() -> None:
    """Malformed JSON arguments fall back to empty dict without raising."""
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_bad",
                "type": "function",
                "function": {"name": "bad_tool", "arguments": "NOT_JSON"},
            }
        ],
    }
    msg = _convert_dict_to_message(raw)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["args"] == {}


def test_convert_dict_to_message_tool_calls_dict_args() -> None:
    """If arguments is already a dict (not a JSON string), it's used directly."""
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "my_tool", "arguments": {"key": "value"}},
            }
        ],
    }
    msg = _convert_dict_to_message(raw)
    assert msg.tool_calls[0]["args"] == {"key": "value"}


def test_convert_dict_to_message_object_style_tool_calls() -> None:
    """Object-style tool call (SDK Pydantic model) is handled via getattr."""
    function_obj = MagicMock()
    function_obj.name = "obj_tool"
    function_obj.arguments = '{"param": 99}'

    tc_obj = MagicMock()
    tc_obj.id = "call_obj_1"
    tc_obj.function = function_obj

    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [tc_obj],  # list of SDK objects, not dicts
    }
    msg = _convert_dict_to_message(raw)

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "obj_tool"
    assert msg.tool_calls[0]["args"] == {"param": 99}
    assert msg.tool_calls[0]["id"] == "call_obj_1"


# ---------------------------------------------------------------------------
# NEW: bind_tools integration
# ---------------------------------------------------------------------------

def test_bind_tools_formats_tools_correctly() -> None:
    """bind_tools converts @tool functions to OpenAI-compatible tool schemas."""
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    llm = ChatSarvam(model="foo")
    bound = llm.bind_tools([get_weather])

    # The bound kwargs should contain 'tools' with the formatted schema
    tools_kwarg = bound.kwargs.get("tools", [])
    assert len(tools_kwarg) == 1
    assert tools_kwarg[0]["type"] == "function"
    assert tools_kwarg[0]["function"]["name"] == "get_weather"


def test_bind_tools_with_tool_choice_string() -> None:
    """tool_choice='auto' is passed through as-is."""
    @tool
    def my_func(x: int) -> int:
        """A simple function."""
        return x

    llm = ChatSarvam(model="foo")
    bound = llm.bind_tools([my_func], tool_choice="auto")
    assert bound.kwargs.get("tool_choice") == "auto"


def test_bind_tools_with_tool_choice_bool_single_tool() -> None:
    """tool_choice=True with one tool resolves to a specific function dict."""
    @tool
    def single_tool(val: str) -> str:
        """A single tool."""
        return val

    llm = ChatSarvam(model="foo")
    bound = llm.bind_tools([single_tool], tool_choice=True)
    tc = bound.kwargs.get("tool_choice")
    assert isinstance(tc, dict)
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "single_tool"


def test_bind_tools_with_tool_choice_bool_multiple_tools_raises() -> None:
    """tool_choice=True with multiple tools raises ValueError."""
    @tool
    def tool_a(x: int) -> int:
        """Tool A."""
        return x

    @tool
    def tool_b(y: str) -> str:
        """Tool B."""
        return y

    llm = ChatSarvam(model="foo")
    with pytest.raises(ValueError, match="tool_choice can only be True when there is one tool"):
        llm.bind_tools([tool_a, tool_b], tool_choice=True)


# ---------------------------------------------------------------------------
# NEW: with_structured_output
# ---------------------------------------------------------------------------

class _SimpleSchema(BaseModel):
    """A simple answer."""
    answer: str = Field(description="The answer")


def test_with_structured_output_function_calling_raises_if_no_schema() -> None:
    """with_structured_output(method='function_calling') requires a schema."""
    llm = ChatSarvam(model="foo")
    with pytest.raises(ValueError, match="schema must be specified"):
        llm.with_structured_output(None, method="function_calling")  # type: ignore[arg-type]


def test_with_structured_output_json_schema_raises_if_no_schema() -> None:
    """with_structured_output(method='json_schema') requires a schema."""
    llm = ChatSarvam(model="foo")
    with pytest.raises(ValueError, match="schema must be specified"):
        llm.with_structured_output(None, method="json_schema")  # type: ignore[arg-type]


def test_with_structured_output_unknown_method_raises() -> None:
    """Unrecognised method string raises ValueError."""
    llm = ChatSarvam(model="foo")
    with pytest.raises(ValueError, match="Unrecognized method"):
        llm.with_structured_output(_SimpleSchema, method="bad_method")  # type: ignore[arg-type]


def test_with_structured_output_function_calling_returns_runnable() -> None:
    """with_structured_output returns a Runnable chain for function_calling."""
    from langchain_core.runnables import RunnableSerializable

    llm = ChatSarvam(model="foo")
    chain = llm.with_structured_output(_SimpleSchema, method="function_calling")
    assert isinstance(chain, RunnableSerializable)


def test_with_structured_output_json_schema_uses_function_calling_internally() -> None:
    """json_schema method falls back to function_calling (tool binding) since
    the Sarvam SDK does not support response_format."""
    llm = ChatSarvam(model="foo")
    chain = llm.with_structured_output(_SimpleSchema, method="json_schema")
    assert chain is not None


def test_with_structured_output_invoke_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: mock the SDK to return a tool call; verify parsed Pydantic object."""
    llm = ChatSarvam(model="foo")

    tool_call_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "foo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "_SimpleSchema",
                                "arguments": json.dumps({"answer": "42"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    mock_client = MagicMock()
    mock_client.completions.return_value = tool_call_response

    chain = llm.with_structured_output(_SimpleSchema, method="function_calling")

    with patch.object(llm, "client", mock_client):
        result = chain.invoke("What is the answer?")

    assert isinstance(result, _SimpleSchema)
    assert result.answer == "42"
