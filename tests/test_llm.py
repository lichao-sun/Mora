

import pytest

from mora.llm.openai_api import OpenAILLM
from mora.configs.llm_config import LLMConfig
test_llm_config = LLMConfig(
    api_key="",
    api_type="",
    model="",


)

@pytest.fixture()
def llm():
    return OpenAILLM(test_llm_config)


@pytest.mark.asyncio
async def test_llm_aask(llm):
    rsp = await llm.aask("hello world", stream=False)

    print(rsp)

    assert len(rsp) > 0





@pytest.mark.asyncio
async def test_llm_acompletion(llm):
    hello_msg = [{"role": "user", "content": "hello"}]
    rsp = await llm.acompletion(hello_msg)
    print(rsp)
    assert len(rsp.choices[0].message.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
