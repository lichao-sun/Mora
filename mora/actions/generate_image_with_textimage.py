import io

import requests
from PIL import Image
import torch
from mora.actions.action import Action
from mora.messages import Message

from diffusers import StableDiffusionInstructPix2PixPipeline


class GenerateImageWithTextAndImage(Action):
    """Generate Image with Text Action"""
    name: str = "Generate Image with Text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                            torch_dtype=torch.float16,
                                                                            safety_checker=None).to("cuda:0")
        print("pix2pix model loaded")

    async def run(self, message: Message):
        """Run action"""

        generator = torch.manual_seed(42)
        input_image = message.image_content
        instruction = message.content

        generate_message = self.model(
            instruction, image=input_image, generator=generator,
        ).images[0]

        return Message(image_content=generate_message)
