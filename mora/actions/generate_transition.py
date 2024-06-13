from mora.actions.action import Action
from mora.messages import Message

from Example_code.SEINE.diffusion import create_diffusion
from Example_code.SEINE.models.unet import UNet3DConditionModel

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from einops import rearrange
from diffusers.models import AutoencoderKL
from mora.actions.SEINE.models.clip import TextEmbedder
from PIL import Image
from torchvision import transforms

from mora.actions.SEINE import video_transforms

from diffusers.utils.import_utils import is_xformers_available


class GenerateTransition(Action):
    """Generate Image with Text Action"""
    name: str = "Generate Image with Text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # device = "cpu"
        self.num_frames=16
        self.use_fp16= True
        self.do_classifier_free_guidance= True
        self.sample_method ="ddpm"
        self.cfg_scale=8.0
        self.use_mask=True
        self.num_sampling_steps=250
        self.device = "cuda:3" if torch.cuda.is_available() else "cpu"
        self.negative_prompt=""
        pretrained_model_path="/home/li0007xu/MoraGen/Mora/model_requirements/SEINE/stable-diffusion-v1-4/"
        ckpt_path = "/home/li0007xu/MoraGen/Mora/model_requirements/SEINE/seine.pt"
        model = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", use_concat=self.use_mask).to(self.device)

        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        # load model 
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['ema']
        model.load_state_dict(state_dict)
        print('loading succeed')

        model.eval()

        diffusion = create_diffusion(str(self.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(self.device)
        text_encoder = TextEmbedder(pretrained_model_path,self.device).to(self.device)

        print('Warnning: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        self.model=model
        self.text_encoder=text_encoder
        self.vae=vae
        self.diffusion=diffusion

    async def run(self, message: Message):

        torch.manual_seed(42)
        torch.set_grad_enabled(False)


        # prompt:
        prompt = 'smooth transition'
        last_frame=message.image_content[0][-1]
        first_frame=message.image_content[1][0]
        init_image_h = last_frame.size[1]
        init_image_w = last_frame.size[0]
        image_h = 512
        image_w = 521
        latent_h = image_h // 8
        latent_w = image_w // 8
        self.image_h = image_h
        self.image_w =  image_w
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.ResizeVideo((image_h, image_w)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        video_frames = []
        transform = transforms.Compose([ 
    transforms.PILToTensor() 
]) 
  
        first_frame=transform(first_frame).unsqueeze(0)
        last_frame=transform(last_frame).unsqueeze(0)
        video_frames.append(last_frame)
        num_zeros = self.num_frames-2

        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            video_frames.append(zeros)

        video_frames.append(first_frame)
        video_frames = torch.cat(video_frames, dim=0) # f,c,h,w
        video_input = self.transform_video(video_frames)


        video_input = video_input.to(self.device).unsqueeze(0) # b,f,c,h,w
        mask = mask_generation_before("onelast1", video_input.shape, video_input.dtype, self.device) # b,f,c,h,w
        masked_video = video_input * (mask == 0)

        video_clip = self.auto_inpainting(video_input, masked_video, mask, prompt, self.vae, self.text_encoder, self.diffusion, self.model, self.device)
        video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        pil_images = []
        for i in range(self.num_frames):
            frame = video_[i]

            # 将 tensor 转换为 numpy 数组
            np_frame = frame.numpy()

            # 创建 PIL 图像
            pil_frame = Image.fromarray(np_frame).resize((init_image_w, init_image_h))
            
            # 将 PIL 图像添加到列表中
            pil_images.append(pil_frame)


        return Message(image_content=message.image_content[0]+pil_images+message.image_content[1])


    def auto_inpainting(self,video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device):
        b,f,c,h,w=video_input.shape
        latent_h = self.image_h // 8
        latent_w = self.image_w // 8

        # prepare inputs
        if self.use_fp16:
            z = torch.randn(1, 4, self.num_frames, self.latent_h, self.latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
            masked_video = masked_video.to(dtype=torch.float16)
            mask = mask.to(dtype=torch.float16)
        else:
            z = torch.randn(1, 4, self.num_frames, self.latent_h, self.latent_w, device=device) # b,c,f,h,w


        masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
        masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
        masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
        mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    
        # classifier_free_guidance
        if self.do_classifier_free_guidance:
            masked_video = torch.cat([masked_video] * 2)
            mask = torch.cat([mask] * 2)
            z = torch.cat([z] * 2)
            prompt_all = [prompt] + [self.negative_prompt]
            
        else:
            masked_video = masked_video
            mask = mask
            z = z
            prompt_all = [prompt]

        text_prompt = text_encoder(text_prompts=prompt_all, train=False)
        model_kwargs = dict(encoder_hidden_states=text_prompt, 
                                class_labels=None, 
                                cfg_scale=self.cfg_scale,
                                use_fp16=self.use_fp16,) # tav unet

        # Sample video:
        if self.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
                mask=mask, x_start=masked_video, use_concat=self.use_mask
            )
        elif self.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
                mask=mask, x_start=masked_video, use_concat=self.use_mask
            )
        samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
        if self.use_fp16:
            samples = samples.to(dtype=torch.float16)

        video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
        video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
        return video_clip



def mask_generation_before(mask_type, shape, dtype, device, dropout_prob=0.0, use_image_num=0):
    b, f, c, h, w = shape
    if mask_type.startswith('first'):
        num = int(mask_type.split('first')[-1])
        mask_f = torch.cat([torch.zeros(1, num, 1, 1, 1, dtype=dtype, device=device),
                           torch.ones(1, f-num, 1, 1, 1, dtype=dtype, device=device)], dim=1)
        mask = mask_f.expand(b, -1, c, h, w)
    elif mask_type.startswith('all'):
        mask = torch.ones(b,f,c,h,w,dtype=dtype,device=device)
    elif mask_type.startswith('onelast'):
        num = int(mask_type.split('onelast')[-1])
        mask_one = torch.zeros(1,1,1,1,1, dtype=dtype, device=device)
        mask_mid = torch.ones(1,f-2*num,1,1,1,dtype=dtype, device=device)
        mask_last = torch.zeros_like(mask_one)
        mask = torch.cat([mask_one]*num + [mask_mid] + [mask_last]*num, dim=1)
        mask = mask.expand(b, -1, c, h, w)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask