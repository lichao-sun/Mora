import asyncio
import os
import sys
from mora.actions import GenerateImageWithTextAndImage
from mora.messages import Message
import pytest
from PIL import Image
import decord

def load_video(video_path):
    frames=[]
    vr = decord.VideoReader(video_path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
    return frames

@pytest.mark.asyncio
async def test_Generate_Image_With_Text_and_Image():
    example_image = Image.open("/home/li0007xu/MoraGen/Mora/tests/test_image_producer.png").convert("RGB")
    msg = Message(content="add some people", image_content=example_image)
    action= GenerateImageWithTextAndImage()
    output_msg = await action.run(msg)
    assert output_msg.image_content is not None
    output_msg.image_content.save("generated.jpg")

if __name__ == '__main__':
    asyncio.run(test_Generate_Image_With_Text_and_Image())