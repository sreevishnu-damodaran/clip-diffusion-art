import argparse
import os
from types import SimpleNamespace
import sys

import cv2
import glob
import numpy as np

import torch
import wandb

from clip_diffusion_art import logger

sys.path.append('./SwinIR')
from SwinIR.main_test_swinir import define_model


class SwinIRPredictor:
    def __init__(self, task, model_path, large_SR=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.window_size = 8
        if task in ['jpeg_car']:
            self.window_size = 7
        
        self.scale = 1
        if task == 'real_sr':
            self.scale = 4

        self.model = define_model(SimpleNamespace(task=task,
                                                  large_model=large_SR,
                                                  model_path=model_path))
        self.model.eval()
        self.model = self.model.to(self.device)


    def predict(self, image):
        image = image.float().to(self.device)
        
        if image.ndim == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = image.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            image = torch.cat([image,
                                torch.flip(image,
                                            [2])], 2)[:, :, :h_old + h_pad, :]
            image = torch.cat([image,
                                torch.flip(image,
                                            [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.model(image)
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        return output.data.squeeze().float().clamp_(0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('data_dir', type=str, nargs='*',
                   help='directory with images')
    p.add_argument('--task', type=str, default='real_sr',
                   choices=['real_sr', 'color_dn', 'gray_dn', 'jpeg_car'],
                   help='image restoration task')
    p.add_argument('--model_path', type=str,
                   help='path to pretrained model weights')
    p.add_argument('--large_sr', type=bool, default=False,
                   help='use large super resolution model')
    p.add_argument('--out_path', type=str, default="out_images",
                   help='directory to store output predictions')
    p.add_argument('--log_dir', type=str,
                   help='logging directory')
    p.add_argument('--wandb_project', type=str,
                   help='wandb project name, specify to integrate wandb logging')
    p.add_argument('--wandb_entity', type=str,
                   help='wandb entity name')
    p.add_argument('--wandb_name', type=str,
                   help='wandb run name')
    args = p.parse_args()

    wandb_run = None
    if args.wandb_project is not None:
        wandb_run = wandb.init(project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=locals())
    else:
        print("--wandb_project not specified. Skipping Wandb integration.")

    logger.configure(dir=args.log_dir, wandb_run=wandb_run)

    logger.log("Creating SR model...")
    swinir = SwinIRPredictor(args.task, args.model_path, args.large_SR)

    if args.task in ['real_sr', 'color_dn']:
        cv2_read_mode = cv2.IMREAD_COLOR
    elif args.task == 'gray_dn':
        cv2_read_mode = cv2.IMREAD_GRAYSCALE
    elif args.task == 'jpeg_car':
        cv2_read_mode = cv2.IMREAD_UNCHANGED

    os.makedirs(args.out_path, exist_ok=True)

    for idx, img_path in enumerate(sorted(glob.glob(os.path.join(args.data_dir, '*')))):
        image = cv2.imread(img_path, cv2_read_mode).astype(np.float32) / 255.

        image = np.transpose(image if image.shape[2] == 1 else image[:, :, [2, 1, 0]],
                                      (2, 0, 1))
 
        image = torch.from_numpy(image)
        output = swinir.predict(image)

        output = output.cpu().numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(args.out_path, os.path.basename(img_path)), output)

    if wandb_run is not None:
        wandb.log({"out_images": [wandb.Image(img_path) for img_path in 
        sorted(glob.glob(os.path.join(args.out_path, '*')))]})

if __name__ == "__main__":
    main()
