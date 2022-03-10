![](https://i.ibb.co/H2sHF0T/cover1-02.jpg)

<br>

# CLIP Diffusion Art

Fine-tune diffusion models on custom datasets and sample with text-conditioning using CLIP guidance and SwinIR for super resolution.

#### 📌 Dataset with public domain artworks created for this project:
 [Artworks in Public Domain](kaggle.com/sreevishnudamodaran/artworks-in-public-domain)
 
#### 📌 Link to interactive run in notebook:
 [Stunning Art with CLIP Guided Diffusion+SwinIR](https://www.kaggle.com/sreevishnudamodaran/stunning-art-with-clip-guided-diffusion-swinir)

📌 Wandb logging is integrated for training and sampling.

<br>

## Generated Samples

<br>
<br>

![](https://i.ibb.co/DpTYvK3/job18-1.gif)
 <br>"vibrant watercolor painting of a flower, artstation HQ"

<br>
<br>

![](https://i.ibb.co/BTWfbf4/23-0.gif)
<br>"beautiful matte painting of dystopian city, Behance HD"

<br>
<br>

![](https://i.ibb.co/8dBTzpX/job18-2.gif)
 <br>"vibrant watercolor painting of a flower, artstation HQ"

<br>
<br>

![](https://i.ibb.co/8gsR0w1/2.gif)
<br>"artstation HQ, photorealistic depiction of an alien city"

<br>

### For more generated artworks, visit this [report](https://wandb.ai/sreevishnu-damodaran/clip_diffusion_art/reports/Results-CLIP-Guided-Diffusion-SwinIR--VmlldzoxNjUxNTMz)

<br>

## Super-resolution Results

<br>
<br>

![](https://i.ibb.co/Gss0y38/sr-zoom-optimized.gif)

<br>
<br>

## Credits

Developed using techniques and architectures borrowed from original work by the authors below:

 - [Guided diffusion](https://github.com/openai/guided-diffusion) and [improved diffusion](https://github.com/openai/improved-diffusion) by [OpenAI](https://github.com/openai)

 - Original notebook on CLIP guidance sampling by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings) with improvements by [nerdyrodent](https://github.com/nerdyrodent/CLIP-Guided-Diffusion) and [sadnow](https://github.com/sadnow/360Diffusion) (@sadly_existent) 

 - SwinIR: Image Restoration Using Shifted Window Transformer from https://github.com/JingyunLiang/SwinIR

Huge thanks to all their great work! I highly recommend checking out these repos.

<br>

## Installation

```
git clone https://github.com/sreevishnu-damodaran/clip-diffusion-art.git -q
cd clip-diffusion-art
pip install -e . -q
git clone https://github.com/JingyunLiang/SwinIR.git -q
git clone https://github.com/crowsonkb/guided-diffusion -q
pip install -e guided-diffusion -q
git clone https://github.com/openai/CLIP -q
pip install -e ./CLIP -q
```

<br>

## Dataset

Public Domain Artworks dataset used in this repo:

https://www.kaggle.com/sreevishnudamodaran/artworks-in-public-domain

Additional details [datasets/README.md](datasets\README.md)

<br>

## Training & Fine-tuning

Chooose the hyperparameters for training. These are resonable defaults to fine-tune on a custom dataset with a 16GB GPUs on Colab or Kaggle:

```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --rescale_learned_sigmas True --rescale_timesteps True --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 5e-6 --save_interval 500 --batch_size 16 --use_fp16 True --wandb_project diffusion-art-train --use_checkpoint True --resume_checkpoint pretrained_models/lsun_uncond_100M_1200K_bs128.pt"
```

Once the hyperparameters are set, run the traning job as follows:

```
python clip_diffusion_art/train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

Refer to the openai [improved diffusion](https://github.com/openai/improved-diffusion) for more details on choosing hyperparameters and to select other pre-trained weights.

<br>


### Download SR pre-trained weights
```
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
```

Passing the `sr_model_path` flag to `sample.py ` performs super-resolution to each image after sampling.

## Sample Images with CLIP Guidance

```
python clip_diffusion_art/sample.py \
"beautiful matte painting of dystopian city, Behance HD" \
--checkpoint 256x256_clip_diffusion_art.pt \
--model_config "clip_diffusion_art/configs/256x256_clip_diffusion_art.yaml" \
--sampling "ddim50" \
--cutn 60 \
--cut_batches 4 \
--sr_model_path pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
--large_sr \
--output_dir "outputs"

```

### Options:
`--images` - image prompts (default=None)<br>
`--checkpoint` - diffusion model checkpoint to use for sampling<br>
`--model_config` - diffusion model config yaml<br>
`--wandb_project` - enable wandb logging and use this project name<br>
`--wandb_name` - optinal run name to use for wandb logging<br>
`--wandb_entity` - optinal entity to use for wandb logging<br>
`--num_samples` - - number of samples to generate (default=1)<br>
`--batch_size` - default=1batch size for the diffusion model<br>
`--sampling` - timestep respacing sampling methods to use (default="ddim50", choices=[25, 50, 100, 150, 250, 500, 1000, ddim25, ddim50, ddim100, ddim150, ddim250, ddim500, ddim1000])<br>
`--diffusion_steps` - number of diffusion timesteps (default=1000)<br>
`--skip_timesteps` - diffusion timesteps to skip (default=5)<br>
`--clip_denoised` - enable to filter out noise from generation (default=False)<br>
`--randomize_class_disable` - disables changing imagenet class randomly in each iteration (default=False)<br>
`--eta` - the amount of noise to add during sampling (default=0)<br>
`--clip_model` - CLIP pre-trained model to use (default="ViT-B/16",
choices=["RN50","RN101","RN50x4","RN50x16","RN50x64","ViT-B/32","ViT-B/16","ViT-L/14"])<br>
`--skip_augs` - enable to skip torchvision augmentations (default=False)<br>
`--cutn` - the number of random crops to use (default=16)<br>
`--cutn_batches` - number of crops to take from the image (default=4)<br>
`--init_image` - init image to use while sampling (default=None)<br>
`--loss_fn` - loss fn to use for CLIP guidance (default="spherical", choices=["spherical" "cos_spherical"])<br>
`--clip_guidance_scale` - CLIP guidance scale (default=5000)<br>
`--tv_scale` - controls smoothing in samples (default=100)<br>
`--range_scale` - controls the range of RGB values in samples (default=150)<br>
`--saturation_scale` - controls the saturation in samples (default=0)<br>
`--init_scale` - controls the adherence to the init image (default=1000)<br>
`--scale_multiplier` - scales clip_guidance_scale tv_scale and range_scale (default=50)<br>
`--disable_grad_clamp` - disable gradient clamping (default=False)<br>
`--sr_model_path` - SwinIR super-resolution model checkpoint (default=None)<br>
`--large_sr` - enable to use large SwinIR super-resolution model (default=False)<br>
`--output_dir` - output images directory (default="output_dir")<br>
`--seed` - the random seed (default=47)<br>
`--device` - the device to use <br>

<br>

## Apply Super-resolution

Use the following to run super-resolution on other images or use it for other tasks (grayscale/color image denoising/JPEG compression artifact reduction)

```
python swinir.py <path-to-images-dir> --task "real_sr"
```

`data_dir` - directory with images

`--task` - image restoration task (default='real_sr', choices=['real_sr', 'color_dn', 'gray_dn', 'jpeg_car'])
