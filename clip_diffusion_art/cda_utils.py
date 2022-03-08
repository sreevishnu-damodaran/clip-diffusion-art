"""
Developed based on code from
_______________________________________

https://github.com/openai/guided-diffusion
https://github.com/openai/improved-diffusion

by OpenAI Team

Please follow the respective code licences

"""

import argparse
import io
import math
import os
import torch
import torchvision.transforms.functional as TF

import urllib
import requests
from tqdm import tqdm
# import urllib.request 


def download_weights(url, out_path):
    response = getattr(urllib, 'request', urllib).urlopen(url)
    filename = url.split('/')[-1]
    with tqdm.wrapattr(open(os.path.join(out_path, filename), "wb"), "write",
                    miniters=1, desc=filename, position = 0, leave=True,
                    total=getattr(response, 'length', None)) as fout:
        for chunk in response:
            fout.write(chunk)

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def pil_to_tensor(x):
    x = TF.to_tensor(x)
    if x.ndim == 2:
        x = x[..., None]
    return x * 2 - 1

def tensor_to_pil(x):
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = torch.squeeze(x, 0)
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])