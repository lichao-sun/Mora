
import gradio as gr

from mora.agent.video_producer_with_text import VideoProducerWithText
from mora.agent.video_producer import VideoProducer
from mora.agent.video_connection import VideoConnection
from mora.agent.video_producer_extension import VideoProducerExtension
from mora.messages import Message
from diffusers.utils import load_image, export_to_video
import decord
from PIL import Image
import asyncio
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
# 加载视频




text_to_video_role = VideoProducer()
image_text_to_video_role= VideoProducerWithText()
video_connection_role= VideoConnection()
video_extension_role=VideoProducerExtension()

def load_video(video_path):
    frames=[]
    vr = decord.VideoReader(video_path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
    return frames

def text_to_video(msg):

    result = asyncio.run(text_to_video_role.run(msg))
    video_path="/home/li0007xu/MoraGen/Mora/generated.mp4"
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames = ImageSequenceClip(video_frames, fps=7)
    video_frames.write_videofile(video_path, codec='libx264')
    text_to_video_role.memory.clear()
    return video_path

def image_video_text_to_video(media, text):
    video=load_video(media)
    msg = Message(content=text, image_content=video[0])

    result = asyncio.run(image_text_to_video_role.run(msg))
    video_path="/home/li0007xu/MoraGen/Mora/generated.mp4"
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames = ImageSequenceClip(video_frames, fps=7)
    video_frames.write_videofile(video_path, codec='libx264')
    image_text_to_video_role.memory.clear()

    return video_path

def video_mixer(video1, video2):

    Video1=load_video(video1)
    Video2=load_video(video2)
    msg1 = Message(content="", image_content=[Video1, Video2])


    result = asyncio.run(video_connection_role.run(msg1))
    video_path="/home/li0007xu/MoraGen/Mora/generated.mp4"
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames = ImageSequenceClip(video_frames, fps=7)
    video_frames.write_videofile(video_path, codec='libx264')
    video_connection_role.memory.clear()
    return video_path

def video_extension(video):
    
    Video=load_video(video)
    msg1 = Message(content="add some people", image_content=Video[-1])

    result = asyncio.run(video_extension_role.run(msg1))
    video_path="/home/li0007xu/MoraGen/Mora/generated.mp4"
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames = ImageSequenceClip(video_frames, fps=7)
    video_frames.write_videofile(video_path, codec='libx264')
    video_extension_role.memory.clear()
    return video_path

def image_to_video(image):
    image=Image.open(image).convert("RGB")
    msg1 = Message(content="add some people", image_content=image)

    result = asyncio.run(video_extension_role.run(msg1))
    video_path="/home/li0007xu/MoraGen/Mora/generated.mp4"
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames = ImageSequenceClip(video_frames, fps=7)
    video_frames.write_videofile(video_path, codec='libx264')
    video_extension_role.memory.clear()
    return video_path

with gr.Blocks() as demo:

    with gr.Tab("Text to Video"):
        with gr.Row():
            text_input = gr.Textbox(label="Enter text")
            submit_text = gr.Button("Generate Video from Text")
        text_output = gr.Video(label="Your video")

    with gr.Tab("Video editing"):
        with gr.Row():
            media_input = gr.File(label="Upload Video", type="filepath")
            text_optional_input = gr.Textbox(label="Enter optional text")
            submit_media = gr.Button("Generate Video from Media")
        media_output = gr.Video(label="Your video")

    with gr.Tab("Video Connection"):
        with gr.Row():
            video1_input = gr.File(label="Upload First Video", type="filepath")
            video2_input = gr.File(label="Upload Second Video", type="filepath")
            submit_videos = gr.Button("Mix Videos")
        mixed_video_output = gr.Video(label="Your mixed video")

    with gr.Tab("Video Extension"):
        with gr.Row():
            video_extension_input = gr.File(label="Upload First Video", type="filepath")
            submit_extension_videos = gr.Button("Extend Videos")
        extened_video_output = gr.Video(label="Your extended video")
    
    with gr.Tab("Image to Video"):
        with gr.Row():
            image_input = gr.File(label="Upload First Image", type="filepath")
            submit_image_videos = gr.Button("Extend Videos")
        video_output = gr.Video(label="Your generated video")

    # Setup interactions
    submit_text.click(text_to_video, inputs=text_input, outputs=text_output)
    submit_media.click(image_video_text_to_video, inputs=[media_input, text_optional_input], outputs=media_output)
    submit_videos.click(video_mixer, inputs=[video1_input, video2_input], outputs=mixed_video_output)
    submit_extension_videos.click(video_extension, inputs=video_extension_input, outputs=extened_video_output)
    submit_image_videos.click(image_to_video, inputs=image_input, outputs=video_output)

demo.launch()