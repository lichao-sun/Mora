import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))   
from diffusers.utils import load_image, export_to_video
from mora.agent.video_producer_with_text import VideoProducerWithText
from mora.messages import Message
from PIL import Image
import asyncio
import decord
def load_video(video_path):
    frames=[]
    vr = decord.VideoReader(video_path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
    return frames
def main(msg="Put the video in space with a rainbow road"):
    # role = SimpleCoder()
    video=load_video("/home/li0007xu/MoraGen/Mora/base.mp4")
    role = VideoProducerWithText()
    # img=Image.open("/home/li0007xu/MoraGen/Mora/input2.jpg")

    msg1=Message(content=msg,image_content=video[0])
    result = asyncio.run(role.run(msg1))
    export_to_video(result.image_content, "generated_edited.mp4", fps=7)
    




if __name__ == "__main__":
    main()