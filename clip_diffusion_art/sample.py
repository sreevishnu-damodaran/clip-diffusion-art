"""
Developed based on code from
_______________________________________

https://github.com/openai/guided-diffusion
by OpenAI Team

Original notebook on CLIP guidance sampling by Katherine Crowson 
(https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
with improvements by [nerdyrodent](https://github.com/nerdyrodent/CLIP-Guided-Diffusion)
 and [sadnow](https://github.com/sadnow/360Diffusion) (@sadly_existent) 

SwinIR: Image Restoration Using Shifted Window Transformer
from https://github.com/JingyunLiang/SwinIR

Please follow the respective code licences


"""

import gc
import sys
import random
import os
import glob
import argparse
from functools import partial

from datetime import datetime
from PIL import Image
from tqdm import tqdm

import numpy as np
import yaml

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import wandb
import lpips

import clip_diffusion_art.cda_utils as cda_utils
from clip_diffusion_art.swinir import SwinIRPredictor

sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')
from CLIP import clip as openaiclip

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults
)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.RandomGrayscale(p=0.15),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def cos_spherical(x, y):
    cos_sim = torch.cosine_similarity(x, y, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    spherical = torch.abs((x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2))
    return spherical + 0.5*(torch.exp((1-cos_sim).div(0.5)))

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class ClipDiffusion:
    def __init__(
        self,
        checkpoint,
        model_config,
        diffusion_steps=1000,
        sampling='ddim50',
        clip_model='ViT-B/16',
        device="cpu"
        ):

        self.device = torch.device(device)
        self.sampling = sampling

        diffusion_config = model_and_diffusion_defaults()
        diffusion_config.update(model_config)
        diffusion_config["diffusion_steps"] = diffusion_steps
        diffusion_config["timestep_respacing"] = self.sampling

        self.model, self.diffusion = create_model_and_diffusion(**diffusion_config)
        self.side_y = self.side_x = diffusion_config['image_size']

        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model = self.model.to(self.device).eval().requires_grad_(False)

        for name, param in self.model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                    param.requires_grad_()
        
        if diffusion_config['use_fp16']:
            self.model.convert_to_fp16()

        self.clip_model, preprocess = openaiclip.load(clip_model, jit=False, device=self.device)
        self.clip_model.eval().requires_grad_(False)
        self.normalize = preprocess.transforms.pop(-1)
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)

        self.clip_size = self.clip_model.visual.input_resolution

        self.init_image = None
        self.make_cutouts = None
        self.target_embeds = None
        self.weights = None
        self.loss_fn = None
        self.upscaler = None
        self.cur_t = None 
        self.cutn = None
        self.cutn_batches = None
        self.clip_guidance_scale = None
        self.tv_scale = None
        self.range_scale = None
        self.saturation_scale = None
        self.init_scale = None
        self.scale_multiplier = None
        self.clamp_grad = None

        self.loss_values = []

    def sample(
        self,
        prompts,
        images=None,
        num_samples=4,
        batch_size=1,
        skip_timesteps=5,
        clip_denoised=False,
        randomize_class=True,
        eta=0.5,
        skip_augs=False,
        cutn=16,
        cutn_batches=4,
        init_image=None,
        loss_fn="spherical",
        clip_guidance_scale=5000,
        tv_scale=100,
        range_scale=150,
        saturation_scale=0,
        init_scale=1000,
        scale_multiplier=1,
        clamp_grad=True,
        output_dir="output_dir",
        wandb_run=None
        ):
        
        self.wandb_run = wandb_run
        self.cutn = cutn
        self.cutn_batches = cutn_batches
        self.clip_guidance_scale = clip_guidance_scale
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.saturation_scale = saturation_scale
        self.init_scale = init_scale
        self.scale_multiplier = scale_multiplier
        self.clamp_grad = clamp_grad
        
        if init_image:
            self.init_image = Image.open(cda_utils.fetch(init_image)).convert('RGB')
            self.init_image = resize_and_center_crop(init_image, (self.side_x, self.side_y))
            self.init_image = cda_utils.pil_to_tensor(init_image).to(self.device)[None]

        self.make_cutouts = MakeCutouts(self.clip_size, cutn, skip_augs=skip_augs)

        self.target_embeds, self.weights = [], []

        for prompt in prompts:
            txt, weight = parse_prompt(prompt)
            self.target_embeds.append(self.clip_model.encode_text(
                openaiclip.tokenize(txt).to(self.device)).float())
            self.weights.append(weight)

        if images is not None:
            for prompt in images:
                path, weight = parse_prompt(prompt)
                img = Image.open(cda_utils.fetch(path)).convert('RGB')
                img = TF.resize(img, min(self.side_x, self.side_y, *img.size),
                                T.InterpolationMode.LANCZOS)
                batch = self.make_cutouts(
                            TF.to_tensor(img)[None].to(self.device))
                embeds = F.normalize(self.clip_model.encode_image(
                    self.normalize(batch)).float(), dim=-1)
                self.target_embeds.append(embeds)
                self.weights.extend([weight / cutn] * self.cutn)

        if not self.target_embeds:
            raise RuntimeError('At least one text or image prompt must be specified.')
        self.target_embeds = torch.cat(self.target_embeds)
        self.weights = torch.tensor(self.weights, device=self.device)
        if self.weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        self.weights /= self.weights.sum().abs()

        if loss_fn == "cos_spherical":
            self.loss_fn = cos_spherical
        else:
            self.loss_fn = spherical_dist_loss

        self.calculate_scale_multiplier()

        self.cur_t = None

        def cond_fn(x, t, y=None, self=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                n = x.shape[0]
                my_t = torch.ones([n], device=self.device,
                                     dtype=torch.long) * self.cur_t
                out = self.diffusion.p_mean_variance(
                    self.model, x, my_t, clip_denoised=False,
                    model_kwargs={'y': y})
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
                
                for i in range(self.cutn_batches):
                    clip_in = self.normalize(self.make_cutouts(x_in.add(1).div(2)))
                    image_embeds = self.clip_model.encode_image(clip_in).float()
                    dists = self.loss_fn(image_embeds.unsqueeze(1),
                     self.target_embeds.unsqueeze(0))
                    dists = dists.view([self.cutn, n, -1])
                    losses = dists.mul(self.weights).sum(2).mean(0)
                    self.loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (torch.autograd.grad(losses.sum() 
                            * self.clip_guidance_scale, x_in)[0] / self.cutn_batches)
                
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                tv_losses = tv_losses.sum() * self.tv_scale
                range_losses = range_losses.sum() * self.range_scale
                sat_losses = sat_losses.sum() * self.saturation_scale
                total_loss = tv_losses + range_losses + sat_losses
                
                if self.init_image is not None and self.init_scale:
                    init_losses = self.lpips_model(x_in, self.init_image)
                    total_loss = (total_loss + init_losses.sum() 
                                    * self.init_scale)
                
                if self.wandb_run is not None:
                    self.wandb_run.log({"clip_loss": self.loss_values[-1],
                        "tv_loss": tv_losses.item(),
                        "sat_losses": sat_losses.item(),
                        "range_losses": range_losses.item(),
                        "total_loss": total_loss.item()
                        })
                
                x_in_grad += torch.autograd.grad(total_loss, x_in)[0]
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]

            if self.clamp_grad:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=0.05) / magnitude
            
            return grad


        if self.sampling.startswith('ddim'):
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.p_sample_loop_progressive

        for i in range(num_samples):
            self.cur_t = self.diffusion.num_timesteps - skip_timesteps - 1

            cond_fn_ = partial(cond_fn, self=self)

            if self.sampling.startswith('ddim'):
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, self.side_y, self.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn_,
                    progress=True,
                    skip_timesteps=skip_timesteps,
                    init_image=self.init_image,
                    randomize_class=randomize_class,
                    eta=eta,
                )
            else:
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, self.side_y, self.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn_,
                    progress=True,
                    skip_timesteps=skip_timesteps,
                    init_image=self.init_image,
                    randomize_class=randomize_class,
                )
            
            for batch in range(batch_size):
                os.makedirs(os.path.join(output_dir,
                                            f"sample{i}_output{batch}_steps"),
                                exist_ok=True)

            with tqdm(samples, unit="step") as tqdm_iter:
                for j, sample in (enumerate(tqdm_iter)):
                    tqdm_iter.set_description(f'Sample {i} ')
                    self.cur_t -= 1

                    for k, image in enumerate(sample['pred_xstart']):
                        pil_image = cda_utils.tensor_to_pil(image)
                        filename = f'output{k}_step{j}.png'
                        pil_image.save(os.path.join(output_dir,
                         f"sample{i}_output{k}_steps", filename))

                    if self.cur_t == -1:
                        gc.collect()
                        torch.cuda.empty_cache()
                        yield image.clamp(-1, 1).add(1).div(2).unsqueeze(0)
                    
                    tqdm_iter.set_postfix(output=k, step=j)

    def upscale(
        self,
        image,
        model_path=None,
        large_sr=True,
        ):
        if self.upscaler is None:
            if model_path is None:
                model_path = "pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
                if not os.path.exists(model_path):
                    print("Downloading and using SwinIR SR Model - realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN")
                    url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
                    cda_utils.download_weights(url, "pretrained_models")

            self.upscaler = SwinIRPredictor('real_sr', model_path, large_sr)
            
        output = self.upscaler.predict(image)

        return output

    def calculate_scale_multiplier(self):
        self.clip_guidance_scale = (self.scale_multiplier 
                                    * self.clip_guidance_scale)
        self.tv_scale = self.scale_multiplier * self.tv_scale
        self.range_scale = self.scale_multiplier * self.range_scale


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompts', type=str, nargs='*',
                   help='text prompts')
    p.add_argument('--images', type=str, nargs='*', default=None,
                   help='image prompts')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='diffusion model checkpoint')
    p.add_argument('--model_config', type=str, default=None,
                   help='diffusion model config yaml')
    p.add_argument('--wandb_project', type=str, default=None,
                   help='enable wandb logging and use this project')
    p.add_argument('--wandb_name', type=str, default=None,
                   help='optinal run name to use for wandb logging')
    p.add_argument('--wandb_entity', type=str, default=None,
                   help='optinal entity to use for wandb logging')
    p.add_argument('--num_samples', type=int, default=1,
                   help='number of samples to generate')
    p.add_argument('--batch_size', type=int, default=1,
                   help='batch size for the diffusion model')
    p.add_argument('--sampling', type=str, default="ddim50", choices=['25','50',
    '100','150','250','500','1000','ddim25','ddim50','ddim100','ddim150',
    'ddim250','ddim500','ddim1000'],
    help='timestep respacing sampling methods to use')
    p.add_argument('--diffusion_steps', type=int, default=1000,
                   help='number of diffusion timesteps')
    p.add_argument('--skip_timesteps', type=int, default=5,
                   help='diffusion timesteps to skip')
    p.add_argument('--clip_denoised', default=False, action="store_true",
                   help='enable to filter out noise from generation')
    p.add_argument('--randomize_class_disable', default=False, action="store_true",
                   help='disables changing imagenet class randomly in each iteration')
    p.add_argument('--eta', type=float, default=0.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--clip_model', type=str, default="ViT-B/16",
                    choices=["RN50","RN101","RN50x4","RN50x16","RN50x64","ViT-B/32","ViT-B/16","ViT-L/14"],
                   help='CLIP pre-trained model to use') 
    p.add_argument('--skip_augs', default=False, action="store_true",
                   help='enable to skip torchvision augmentations')
    p.add_argument('--cutn', type=int, default=30,
                   help='the number of random crops to use')
    p.add_argument('--cutn_batches', type=int, default=4,
                   help='number of crops to take from the image')
    p.add_argument('--init_image', type=str, default=None,
                   help='init image to use  while sampling')
    p.add_argument('--loss_fn', type=str, default="spherical",
                    choices=["spherical", "cos_spherical"],
                   help='loss fn to use for CLIP guidance')
    p.add_argument('--clip_guidance_scale', type=int, default=5000,
                   help='CLIP guidance scale')
    p.add_argument('--tv_scale', type=int, default=100,
                   help='controls smoothing in samples')
    p.add_argument('--range_scale', type=int, default=150,
                   help='controls the range of RGB values in samples')
    p.add_argument('--saturation_scale', type=int, default=0,
                   help='controls the saturation in samples')
    p.add_argument('--init_scale', type=int, default=1000,
                   help='controls the adherence to the init image')
    p.add_argument('--scale_multiplier', type=int, default=50,
                   help='scales clip_guidance_scale, tv_scale and range_scale')
    p.add_argument('--disable_grad_clamp', default=False, action="store_true",
                   help='disable gradient clamping')
    p.add_argument('--sr_model_path', type=str, default=None,
                   help='SwinIR super-resolution model checkpoint')
    p.add_argument('--large_sr', default=False, action="store_true",
                   help='enable to use large SwinIR super-resolution model')
    p.add_argument('--output_dir', type=str, default="output_dir",
                   help='output images directory')
    p.add_argument('--seed', type=int, default=47,
                   help='the random seed')
    p.add_argument('--device', type=str, default=None,
                   help='the device to use')
    
    args = p.parse_args()

    wandb_run = None
    if args.wandb_project is not None:
        wandb_run = wandb.init(project=args.wandb_project,
                                    entity=args.wandb_entity,
                                    name=args.wandb_name)    
    else:
        print(f"--wandb_project not specified. Skipping Wandb integration.")

    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print('Using device:', device)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    config_file = open(args.model_config)

    model_config = yaml.load(config_file,
                                 Loader=yaml.FullLoader)["model_config"]
    print("\nmodel_config", model_config)

    if args.checkpoint is None:
        checkpoint = "pretrained_models/256x256_clip_diffusion_art.pt"
        if not os.path.exists(checkpoint):
            url = "https://api.wandb.ai/files/sreevishnu-damodaran/clip_diffusion_art/29bag3br/256x256_clip_diffusion_art.pt"
            cda_utils.download_weights(url, "pretrained_models")        
    else:
        checkpoint = args.checkpoint

    clip_diffusion = ClipDiffusion(checkpoint,
        model_config=model_config,
        sampling=args.sampling,
        diffusion_steps=args.diffusion_steps,
        clip_model=args.clip_model,
        device=device
    )
    
    out_generator = clip_diffusion.sample(
                        args.prompts,
                        images=args.images,
                        num_samples=args.num_samples,
                        skip_timesteps=args.skip_timesteps,
                        clip_denoised=args.clip_denoised,
                        randomize_class=(not args.randomize_class_disable),
                        eta=args.eta,
                        skip_augs=args.skip_augs,
                        cutn=args.cutn,
                        cutn_batches=args.cutn_batches,
                        init_image=args.init_image,
                        loss_fn=args.loss_fn,
                        clip_guidance_scale=args.clip_guidance_scale,
                        tv_scale=args.tv_scale,
                        range_scale=args.range_scale,
                        saturation_scale=args.saturation_scale,
                        init_scale=args.init_scale,
                        scale_multiplier=args.scale_multiplier,
                        clamp_grad=(not args.disable_grad_clamp),
                        output_dir=args.output_dir,
                        wandb_run=wandb_run
                    )


    os.makedirs(args.output_dir, exist_ok=True)

    for i, out_image in enumerate(out_generator):
         # SwinIR Upscaling
        if args.sr_model_path:
            print("\nUpscaling generated image using SwinIR SR...")
            out_image = clip_diffusion.upscale(
                                            out_image,
                                            args.sr_model_path,
                                            large_sr=args.large_sr
                                        )
        
        out_image = TF.to_pil_image(out_image.squeeze(0))
        current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
        filename = f'image{i}_{current_time}.png'
        out_image.save(os.path.join(args.output_dir, filename))

        if wandb_run is not None:
            wandb.log({os.path.splitext(filename)[0]: wandb.Image(
                os.path.join(args.output_dir, filename))})

    if wandb_run is not None:
        for k in range(args.batch_size):
            for i in range(args.num_samples):
                img_files = glob.glob(os.path.join(args.output_dir,
                                    f"sample{i}_output{k}_steps", '*'))
                wandb.log(
                {f"sample{i}_output{k}": [wandb.Image(img_path) for img_path
                in sorted(img_files,
                          key=lambda x: int(os.path.splitext(x)[0]
                          .split("_")[-1].lstrip("step")))]}
                )

    wandb.finish()

if __name__ == '__main__':
    main()