from mora.llm.ollama_api import OllamaLLM
from mora.configs.llm_config import LLMConfig
test_llm_config = LLMConfig(
    api_type="ollama",
    base_url= "http://127.0.0.1:11434/api",
    model="llama3",

)

import pytest
@pytest.fixture()
def llm():
    return OllamaLLM(test_llm_config)


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
