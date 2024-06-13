# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
@Modified By: mashenquan, 2023/11/21. Fix bug: ReadTimeout.
@Modified By: mashenquan, 2023/12/1. Fix bug: Unclosed connection caused by openai 0.x.
"""
from __future__ import annotations

import json
import re

from openai import APIConnectionError, AsyncOpenAI, AsyncStream
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from mora.llm.llm import BaseLLM
from mora.configs.llm_config import LLMConfig

LLM_API_TIMEOUT = 300
USE_CONFIG_TIMEOUT = 0


class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
        self.auto_max_tokens = False

    def _init_client(self):
        """https://github.com/openai/openai-python#async-usage"""
        self.model = self.config.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxies": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout)), stream=True
        )
        usage = None
        collected_messages = []
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
            finish_reason = (
                chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
            )

            collected_messages.append(chunk_message)
            if finish_reason:
                if hasattr(chunk, "usage"):
                    # Some services have usage as an attribute of the chunk, such as Fireworks
                    usage = CompletionUsage(**chunk.usage)
                elif hasattr(chunk.choices[0], "usage"):
                    # The usage of some services is an attribute of chunk.choices[0], such as Moonshot
                    usage = CompletionUsage(**chunk.choices[0].usage)

        full_reply_content = "".join(collected_messages)

        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": 4096,
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            # "stop": None,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)

        return rsp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),

        retry=retry_if_exception_type(APIConnectionError),

    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout))
        return self.get_choice_text(rsp)

    async def _achat_completion_function(
            self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT, **chat_configs
    ) -> ChatCompletion:
        messages = self.format_msg(messages)
        kwargs = self._cons_kwargs(messages=messages, timeout=self.get_timeout(timeout), **chat_configs)
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)

        return rsp

    def _parse_arguments(self, arguments: str) -> dict:
        """parse arguments in openai function call"""
        if "language" not in arguments and "code" not in arguments:
            return {"language": "python", "code": arguments}

        # 匹配language
        language_pattern = re.compile(r'[\"\']?language[\"\']?\s*:\s*["\']([^"\']+?)["\']', re.DOTALL)
        language_match = language_pattern.search(arguments)
        language_value = language_match.group(1) if language_match else "python"

        # 匹配code
        code_pattern = r'(["\'`]{3}|["\'`])([\s\S]*?)\1'
        try:
            code_value = re.findall(code_pattern, arguments)[-1][-1]
        except Exception as e:

            code_value = None

        if code_value is None:
            raise ValueError(f"Parse code error for {arguments}")
        # arguments只有code的情况
        return {"language": language_value, "code": code_value}

    # @handle_exception
    def get_choice_function_arguments(self, rsp: ChatCompletion) -> dict:
        """Required to provide the first function arguments of choice.

        :param dict rsp: same as in self.get_choice_function(rsp)
        :return dict: return the first function arguments of choice, for example,
            {'language': 'python', 'code': "print('Hello, World!')"}
        """
        message = rsp.choices[0].message
        if (
                message.tool_calls is not None
                and message.tool_calls[0].function is not None
                and message.tool_calls[0].function.arguments is not None
        ):
            # reponse is code
            try:
                return json.loads(message.tool_calls[0].function.arguments, strict=False)
            except json.decoder.JSONDecodeError as e:
                error_msg = (
                    f"Got JSONDecodeError for \n{'--' * 40} \n{message.tool_calls[0].function.arguments}, {str(e)}"
                )

                return self._parse_arguments(message.tool_calls[0].function.arguments)
        elif message.tool_calls is None and message.content is not None:
            # reponse is code, fix openai tools_call respond bug,
            # The response content is `code``, but it appears in the content instead of the arguments.
            code_formats = "```"

            # reponse is message
            return {"language": "markdown", "code": self.get_choice_text(rsp)}
        else:

            raise Exception(f"Failed to parse \n {rsp}\n")

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""

    async def atext_to_speech(self, **kwargs):
        """text to speech"""
        return await self.aclient.audio.speech.create(**kwargs)

    async def aspeech_to_text(self, **kwargs):
        """speech to text"""
        return await self.aclient.audio.transcriptions.create(**kwargs)
