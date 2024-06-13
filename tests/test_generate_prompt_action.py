
from mora.actions import GeneratePrompt
from mora.messages import Message
import pytest


@pytest.mark.asyncio
async def test_g():
    msg = Message(content="a man in a suit with a hat and a cane at a train station.")
    action= GeneratePrompt()
    output_msg = await action.run(msg)
    assert output_msg.content is not None
    print(output_msg.content)
