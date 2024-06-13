import io

import requests
from PIL import Image

from mora.actions.action import Action
from mora.messages import Message
from diffusers import DiffusionPipeline
import torch
# API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# headers = {"Authorization": "Bearer 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"}


class GenerateImageWithText(Action):
    """Generate Image with Text Action"""
    name: str = "Generate Image with Text"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
        base.to("cuda:1")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to("cuda:1")
        self.base=base
        self.refiner=refiner

# Define how many steps and what % of steps to be run on each experts (80/20) here
        self.n_steps = 40
        self.high_noise_frac = 0.8
    async def run(self, message: Message):
        prompt=message.content
        image = self.base(
        prompt=prompt,
        num_inference_steps=self.n_steps,
        denoising_end=self.high_noise_frac,
        height=576,
        width=1024,
        output_type="latent",
    ).images
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            height=576,
            width=1024,
            image=image,
        ).images[0]
        return Message(image_content=image)
    #     """Run action"""
    #     payload = {
    #         "inputs": message.content,
    #     }
    #     response = requests.post(API_URL, headers=headers, json=payload)
    #     image_bytes = response.content

    #     image = Image.open(io.BytesIO(image_bytes))
    #     return Message(image_content=image)


