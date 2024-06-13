

import pytest

from mora.llm.azure_openai_api import AzureOpenAILLM
from mora.configs.llm_config import LLMConfig
test_llm_config = LLMConfig(
    api_key='',
    api_version="",
    azure_endpoint="",
    model=''

)
@pytest.fixture()
def llm():
    return AzureOpenAILLM(test_llm_config)


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