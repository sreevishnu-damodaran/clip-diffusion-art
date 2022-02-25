### Download Diffusion models and SwinIR SR models checkpoints to this directory

## Diffusion Model Weights Pre-trained on Public Artworks Dataset

```
wget https://api.wandb.ai/files/sreevishnu-damodaran/clip_diffusion_art/29bag3br/256x256_clip_diffusion_art.pt -P ./pretrained_models -q
```

#### Original Improved Diffusion Pre-trained Weights

##### LSUN bedroom model (lr=1e-4):
```
wget https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_1200K_bs128.pt -P ./pretrained_models -q
```

##### LSUN bedroom model (lr=2e-5):
```
wget https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_2400K_bs64.pt -P ./pretrained_models -q
```


#### SwinIR Pre-trained Weights

##### Real-World Image Super-Resolution:
```
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P ./pretrained_models -q
```

##### Real-World Image Super-Resolution Large:
```
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth -P ./pretrained_models -q
```