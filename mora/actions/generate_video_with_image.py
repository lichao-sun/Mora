import io

import requests
from PIL import Image
import torch
from mora.actions.action import Action
from mora.messages import Message

from diffusers import StableVideoDiffusionPipeline


class GenerateVideoWithImage(Action):
    """Generate Image with Text Action"""
    name: str = "Generate Image with Text"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.model= StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16,variant='fp16').to("cuda:2")
        self.model.enable_model_cpu_offload()
        print("SVD model loaded")
    async def run(self, message: Message):
        """Run action"""


        generator = torch.manual_seed(42)
        current_image = message.image_content
        current_image = current_image.resize((1024,576))
        generate_message=[]
        for iteration in range(3):
            frames =self.model(current_image, decode_chunk_size=12, generator=generator,motion_bucket_id=180, noise_aug_strength=0.1).frames[0]


            generate_message+=frames

            # Get the last frame of the current video for the next iterationRGB
            current_image = frames[-1]


        return Message(image_content=generate_message)
