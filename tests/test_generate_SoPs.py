
from mora.actions import GenerateSoPs
from mora.messages import Message
import pytest


@pytest.mark.asyncio
async def test_g():
    msg = Message(content=str(["GeneratePrompt", "GenerateImageWithTextAPI", "GenerateVideoWithImage"]))
    goal="Generate prompt, and generate image with the generated prompt,then generate video with the generated image"
    action= GenerateSoPs()
    output_msg = await action.run(msg,goal,n_actions=3)

    print(output_msg)
