"""Sarvam chat models."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, Union, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableMap

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.utils import _build_model_kwargs
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


_STRICT_STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        "sarvam/sarvam-30b",
        "sarvam/sarvam-120b",
    }
)

# Parameters accepted by the Sarvam SDK's completions() method.
# Any kwargs not in this set are filtered out before calling the SDK.
_SARVAM_SDK_PARAMS = frozenset(
    {
        "model",
        "temperature",
        "top_p",
        "reasoning_effort",
        "max_tokens",
        "stream",
        "stop",
        "n",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "wiki_grounding",
        "tools",
        "tool_choice",
        "request_options",
    }
)


class ChatSarvam(BaseChatModel):
    """Sarvam AI chat model integration for LangChain.

    Sarvam AI provides multilingual AI models with native support for 10+ 
    Indic languages including Hindi, Bengali, Telugu, Tamil, and more.


    Setup:
        Install `langchain-sarvam` and set environment variable
        `SARVAM_API_KEY`.

        ```python

            pip install -U langchain-sarvam
            export SARVAM_API_KEY="your-api-key"

        ```

    Key init args — completion params:
        model_name: 
            Model name to use. e.g. "sarvam-m".
        temperature: 
            Sampling temperature between 0.0 and 1.0. 
        top_p: 
            Nucleus sampling parameter. Defaults to 1.0.
        reasoning_effort: 
            Reasoning effort level. One of "low", "medium", "high".
        max_tokens: 
           Max number of tokens to generate.
        stop: 
            Stop sequences. Can be a string or list of strings.
        n: 
            Number of completions to generate. Optional (1-128). Defaults to 1.
        frequency_penalty: 
            Penalize frequent tokens. Defaults to None. (-2.0 to 2.0)
        presence_penalty: 
            Penalize new tokens. Defaults to None. (-2.0 to 2.0)
        seed: 
            Random seed for reproducibility.
        wiki_grounding: 
            Enable wiki grounding. Defaults to False.

    Key init args — client params:
        sarvam_api_key: 
            Sarvam AI API key. If not passed in will be read from env var SARVAM_API_KEY.
        request_timeout: 
            Request timeout in seconds.
        streaming: 
            Whether to stream responses. Defaults to False.
        http_client: 
            Custom HTTP client for sync requests.
        http_async_client: 
            Custom HTTP client for async requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python

            from langchain_sarvam import ChatSarvam

            llm = ChatSarvam(
                model_name="sarvam-m",
                temperature=0.7,
                max_tokens=256,
                # other params
            )
        ```

    Invoke:
        ```python

            messages = [
                ("system", "You are a helpful assistant that speaks Hindi."),
                ("human", "What is the color of the sky?"),
            ]
            llm.invoke(messages)
        ```


        ```python

            AIMessage(content='आसमान का रंग नीला होता है।', response_metadata={...})
        ```

    Stream:
        ```python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)
        ```

    Async:
        ```python

            await llm.ainvoke(messages)

        ```
        
        ```python

            async for chunk in llm.astream(messages):
                print(chunk.content, end="", flush=True)
        ```

    Batch:
        ```python

            llm.batch([messages1, messages2])
        ```
        

    Multilingual support:
        Sarvam AI natively supports 10+ Indic languages:

        ```python

            # Hindi
            messages = [
                ("system", "talk in Hindi"),
                ("human", "Hello, how are you?"),
            ]
            response = llm.invoke(messages)
            print(response.content)  # Output in Hindi
            
        ```

    Response metadata:
        ```python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata
        ```
        
        ```python

            {
                'token_usage': {'completion_tokens': 12, 'prompt_tokens': 57, 'total_tokens': 69},
                'model_name': 'sarvam-m',
                'finish_reason': 'stop',
            }
        ```
"""


    # Client instances (internal use only)
    client: Any = Field(
        default=None,
        exclude=True,
        description="Internal Sarvam AI synchronous client instance.",
    )
    async_client: Any = Field(
        default=None,
        exclude=True,
        description="Internal Sarvam AI asynchronous client instance.",
    )


    # Model parameters
    model_name: str = Field(
        alias="model",
        description="Model name to use for chat completions. Defaults to 'sarvam-m'.",
    )

    @property
    def model(self) -> str:
        """same as model_name"""
        return self.model_name

    temperature: float | None = Field(
        default=0.2,
        description="Sampling temperature between 0.0 and 1.0. Higher values make output more random.",
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter. Controls diversity via cumulative probability.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate. If None, uses model's default maximum.",
    )
    n: int = Field(
        default=1,
        description="Number of completions to generate. Must be 1 when streaming is enabled.",
    )
    stop: list[str] | str | None = Field(
        default=None,
        alias="stop_sequences",
        description="Stop sequences. Can be a string or list of strings.",
    )

    # Advanced parameters
    frequency_penalty: float | None = Field(
        default=None,
        description="Penalizes frequent tokens to reduce repetition. Values between -2.0 and 2.0.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Penalizes new tokens to encourage topic diversity. Values between -2.0 and 2.0.",
    )
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort level for the model."
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducible outputs."
    )
    wiki_grounding: bool | None = Field(
        default=False, description="Enable wiki grounding for factual responses."
    )

    # Additional model kwargs
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to the Sarvam AI API.",
    )

    # Authentication and client configuration
    sarvam_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("SARVAM_API_KEY", default=None),
        description="Sarvam AI API key. If not provided, reads from SARVAM_API_KEY environment variable.",
    )
    request_timeout: float | None = Field(
        default=None, alias="timeout", description="Request timeout in seconds."
    )

    # HTTP client customization
    http_client: Any | None = Field(
        default=None, description="Custom HTTP client for synchronous requests."
    )
    http_async_client: Any | None = Field(
        default=None, description="Custom HTTP client for asynchronous requests."
    )

    # Streaming configuration
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses. When True, enables real-time token streaming.",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """
        Build "model_kwargs" param from Pydantic constructor values.

        Args:
            values: All init args passed in by user.
            all_required_field_names: All required field names for the pydantic class.

        Returns:
            dict[str, Any]: Extra kwargs.

        Raises:
            ValueError: If a field is specified in both values and extra_kwargs.
            ValueError: If a field is specified in model_kwargs.
        """
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate the environment and initialize clients.

        This method performs validation of model parameters and initializes
        the Sarvam AI client instances for both sync and async operations.

        Returns:
            Self: The validated model instance.

        Raises:
            ValueError: If n < 1 or if streaming is enabled with n > 1.
            ImportError: If the sarvamai package is not installed.
            ValueError: If the API key is not provided.
        """
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        try:
            from sarvamai import AsyncSarvamAI, SarvamAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import sarvamai python package. Please install it with `pip install sarvamai`."
            ) from exc

        client_params: dict[str, Any] = {
            "api_subscription_key": (
                self.sarvam_api_key.get_secret_value() if self.sarvam_api_key else None
            ),
            "timeout": self.request_timeout,
        }

        if client_params["api_subscription_key"] is None:
            raise ValueError(
                "Sarvam API key is not set. Set `sarvam_api_key` field or `SARVAM_API_KEY` env var."
            )

        if not self.client:
            sync_specific: dict[str, Any] = {}
            if self.http_client is not None:
                sync_specific["httpx_client"] = self.http_client
            self.client = SarvamAI(**client_params, **sync_specific).chat
        if not self.async_client:
            async_specific: dict[str, Any] = {}
            if self.http_async_client is not None:
                async_specific["httpx_client"] = self.http_async_client
            self.async_client = AsyncSarvamAI(**client_params, **async_specific).chat
        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return the secret environment variable names for this model.

        Returns:
            Dictionary mapping field names to environment variable names.
        """
        return {"sarvam_api_key": "SARVAM_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain.

        Returns:
            True, as ChatSarvam supports serialization.
        """
        return True

    @property
    def _llm_type(self) -> str:
        """Return the type identifier for this LLM.

        Returns:
            String identifier for LangChain's internal use.
        """
        return "sarvam-chat"

    def _get_ls_params(
        self, 
        stop: list[str] | None = None, 
        **kwargs: Any
    ) -> LangSmithParams:
        """Get parameters for LangSmith tracing.

        Args:
            stop: Stop sequences to override the instance default.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            LangSmithParams object with tracing information.
        """ 
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="sarvam",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )

        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for API calls.

        Returns:
            Dictionary of parameters to send to the Sarvam AI API.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "n": self.n,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            params["seed"] = self.seed
        if self.wiki_grounding is not None:
            params["wiki_grounding"] = self.wiki_grounding
        if self.model_kwargs:
            params.update(self.model_kwargs)
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Create message dictionaries and parameters for API calls.

        Args:
            messages: List of BaseMessage objects to convert.
            stop: Stop sequences to override instance defaults.

        Returns:
            Tuple of (message_dicts, params) for the API call.
        """
        params = self._default_params()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion synchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the generated response.
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        # Filter out kwargs not supported by the Sarvam SDK
        sdk_params = {k: v for k, v in params.items() if k in _SARVAM_SDK_PARAMS}
        resp = self.client.completions(messages=message_dicts, **sdk_params)
        return self._create_chat_result(resp, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion asynchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Async callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the generated response.
        """
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        # Filter out kwargs not supported by the Sarvam SDK
        sdk_params = {k: v for k, v in params.items() if k in _SARVAM_SDK_PARAMS}
        resp = await self.async_client.completions(messages=message_dicts, **sdk_params)
        return self._create_chat_result(resp, params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion responses synchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk objects as they become available.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.completions(messages=message_dicts, **params):
            processed_chunk = chunk
            if not isinstance(processed_chunk, dict):
                processed_chunk = processed_chunk.model_dump()  # type: ignore[attr-defined]
            if len(processed_chunk.get("choices", [])) == 0:
                continue
            choice = processed_chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                processed_chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := processed_chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := processed_chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat completion responses asynchronously.

        Args:
            messages: List of messages for the conversation.
            stop: Optional stop sequences.
            run_manager: Async callback manager for run tracking.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk objects as they become available.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in self.async_client.completions(
            messages=message_dicts, **params
        ):
            processed_chunk = chunk
            if not isinstance(processed_chunk, dict):
                processed_chunk = processed_chunk.model_dump()  # type: ignore[attr-defined]
            if len(processed_chunk.get("choices", [])) == 0:
                continue
            choice = processed_chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                processed_chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := processed_chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := processed_chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _create_usage_metadata(self, sarvam_token_usage: dict) -> UsageMetadata:
        """
        Create usage metadata from Sarvam token usage response.

        Args:
            sarvam_token_usage: Token usage dict from Sarvam API response.

        Returns:
            Usage metadata dict with input/output token details.
        """

        input_tokens = (
            sarvam_token_usage.get("input_token") or
            sarvam_token_usage.get("prompt_tokens") or 
            0
        )

        output_tokens = (
            sarvam_token_usage.get("output_token") or
            sarvam_token_usage.get("completion_tokens") or 
            0
        )

        total_tokens = (
            sarvam_token_usage.get("total_tokens") or
            input_tokens + output_tokens
        )

        # Support both formats for token details:
        # Responses API uses "*_tokens_details", Chat Completions API might use
        # "prompt_token_details"

        # input_details_dict = (
        #     sarvam_token_usage.get("input_tokens_details")
        #     or sarvam_token_usage.get("prompt_tokens_details")
        #     or {}
        # )
        # output_details_dict = (
        #     sarvam_token_usage.get("output_tokens_details")
        #     or sarvam_token_usage.get("completion_tokens_details")
        #     or {}
        # )

        usage_metadata: UsageMetadata ={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # "input_token_details": input_details_dict,
            # "output_token_details": output_details_dict,
        }

        return usage_metadata

    def _create_chat_result(
        self, response: dict | BaseModel, params: Mapping[str, Any]
    ) -> ChatResult:
        """Create a ChatResult from the API response.

        Args:
            response: Raw response from the Sarvam AI API.
            params: Parameters used for the request (unused but required by interface).

        Returns:
            ChatResult containing the processed response.
        """
        # params is unused but required by base class interface
        generations: list[ChatGeneration] = []
        if not isinstance(response, dict):
            response = response.model_dump()  # type: ignore[attr-defined]
        token_usage = response.get("usage", {})
        for res in response.get("choices", []):
            message = _convert_dict_to_message(res["message"])  # type: ignore[index]
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = self._create_usage_metadata(token_usage)
            generation_info: dict[str, Any] = {
                "finish_reason": res.get("finish_reason")
            }
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model_name" : self.model_name,
            "system_fingerprint": response.get("system_fingerprint")
        }
        reasoning_effort = params.get("reasoning_effort") or self.reasoning_effort
        if reasoning_effort:
            llm_output['reasoning_effort'] = reasoning_effort
        return ChatResult(generations=generations, llm_output=llm_output or None)


    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports Sarvam format tool schemas and any tool definition handled
                by `convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                'auto' to automatically determine which function to call.
            **kwargs: Any additional parameters to pass to the Runnable constructor.
        """
        _ = kwargs.pop("strict", None)

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be a Pydantic class, TypedDict, JSON Schema
                dict, or OpenAI tool schema.
            method: The method to use for structured output.
                - "function_calling": Use function calling (default, recommended).
                - "json_mode": Use JSON mode.
                - "json_schema": Falls back to function_calling internally since the
                  Sarvam SDK does not support response_format natively.
            include_raw: If True, return both raw and parsed outputs.
            strict: Only used with json_schema; ignored for unsupported models.
            **kwargs: Additional parameters for the Runnable constructor.
        """
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
                **kwargs,
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            # The Sarvam SDK does not support the response_format parameter,
            # so json_schema cannot work natively. Fall back to function_calling
            # which is reliably supported and produces equivalent structured output.
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": formatted_tool,
                },
                **kwargs,
            )
            if is_pydantic_schema:
                output_parser = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
                **kwargs,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                "Unrecognized method argument. Expected one of "
                "'function_calling', 'json_mode', or 'json_schema'. "
                f"Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        return {
            "role": "user", 
            "content": message.content
        }
    elif isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, list):
            text_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = text_blocks if text_blocks else ""
        return {"role": "assistant", "content": content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        return {
            "role": "function",
            "content": message.content, 
            "name": message.name
        }
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict

def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    delta = cast("Mapping[str, Any]", choice.get("delta", {}))
    role = cast("str | None", delta.get("role"))
    content = cast("str", delta.get("content") or "")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=delta.get("name"))  # type: ignore[arg-type]
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=delta.get("tool_call_id"))  # type: ignore[arg-type]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.

    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        tool_calls_data = _dict.get("tool_calls")
        tool_calls = []
        if tool_calls_data:
            additional_kwargs["tool_calls"] = tool_calls_data
            for tc in tool_calls_data:
                # Handle both dict and object-style tool call data
                if isinstance(tc, dict):
                    function_info = tc.get("function", {})
                    tc_id = tc.get("id", "")
                    fn_name = function_info.get("name", "")
                    fn_args = function_info.get("arguments", "")
                else:
                    # Object-style (e.g. Pydantic model from SDK)
                    function_info = getattr(tc, "function", None)
                    tc_id = getattr(tc, "id", "") or ""
                    fn_name = getattr(function_info, "name", "") if function_info else ""
                    fn_args = getattr(function_info, "arguments", "") if function_info else ""

                # Parse arguments from JSON string if needed
                if isinstance(fn_args, str):
                    try:
                        parsed_args = json.loads(fn_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                else:
                    parsed_args = fn_args if isinstance(fn_args, dict) else {}

                tool_calls.append(
                    {
                        "name": fn_name,
                        "args": parsed_args,
                        "id": tc_id,
                        "type": "tool_call",
                    }
                )

        return AIMessage(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            response_metadata={"model_provider": "sarvam"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    if role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""), tool_call_id=_dict.get("tool_call_id")
        )  # type: ignore[arg-type]
    return ChatMessage(content=_dict.get("content", ""), role=cast("str", role))
