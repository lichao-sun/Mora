import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .unet import UNet3DConditionModel
from torch.optim.lr_scheduler import LambdaLR

def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'UNet' in args.model:
        pretrained_model_path = args.pretrained_model_path
        return UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", use_concat=args.use_mask)
    else:
        raise '{} Model Not Supported!'.format(args.model)
    