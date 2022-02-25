# Datasets

## Public Domain Artworks dataset used in this repo:

https://www.kaggle.com/sreevishnudamodaran/artworks-in-public-domain

<br>

#### Number of images:<br>
29.3k<br>

#### Size:<br>
256x256

<br>

All artworks were scraped from the [WikiArt](https://www.wikiart.org/) and [rawpixel](https://www.rawpixel.com/) from their open collection of artworks in the public domain.

<br>

To use custom datasets for training, download/scrape the necessary images and then resize them (and preferably center crop to avoid aspect ratio change) to the input size of the diffusion model of choice.

Make sure all the images have 3 dimensions (RGB) incase of grayscale images.

Training is as simple as putting them all into a directory which can be passed to the training scripts via the `--data_dir` argument.
