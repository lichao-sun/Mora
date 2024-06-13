import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mora.actions import GenerateTransition
from mora.messages import Message
import pytest
from PIL import Image
import decord
from diffusers.utils import export_to_video
import numpy as np
def load_video(video_path):
    frames=[]
    vr = decord.VideoReader(video_path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
    return frames

@pytest.mark.asyncio
async def test_generate_transition():
    video_path="/home/li0007xu/MoraGen/Mora/tests/generated.mp4"
    Video1=[Image.open("/home/li0007xu/MoraGen/Mora/SEINE/input/transition/2/1.png")]
    Video2=[Image.open("/home/li0007xu/MoraGen/Mora/SEINE/input/transition/2/2.png")]
    # Video1=load_video("/home/li0007xu/MoraGen/Mora/base.mp4")
    # Video2=load_video("/home/li0007xu/MoraGen/Mora/base.mp4")
    msg1 = Message(content="", image_content=[Video1,Video2])

    action= GenerateTransition()
    output_msg = await action.run(msg1)
    export_to_video(output_msg.image_content, "newgenerated.mp4", fps=7)


if __name__ == '__main__':
    asyncio.run(test_generate_transition())