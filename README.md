# Latent Diffusion Models

This repository contains modified version of [Latent Diffusion Models](https://github.com/Stability-AI/stablediffusion/tree/main/ldm) library pocket.
________________________________

## Requirements

You can update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 
#### xformers efficient attention
For more efficiency and speed on GPUs, 
we highly recommended installing the [xformers](https://github.com/facebookresearch/xformers)
library.

Tested on A100 with CUDA 11.4.
Installation needs a somewhat recent version of nvcc and gcc/g++, obtain those, e.g., via 
```commandline
export CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64==9.5.0
```

Then, run the following (compiling takes up to 30 min).

```commandline
cd ..
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ../stablediffusion
```
Upon successful installation, the code will automatically default to [memory efficient attention](https://github.com/facebookresearch/xformers)
for the self- and cross-attention layers in the U-Net and autoencoder.

## Install

```
pip install git+https://github.com/StableDraw/ldm
``` 
or for locale build:
```
python setup.py build
python bdist_wheel
``` 
then your wheel will be in the dist directory, then you may install it, for example, by
```
cd dist
pip install ldm-2.1.1-py3-none-any.whl
``` 
or with another name of your package that will be in dist directory

## General Disclaimer
Stable Diffusion models are general text-to-image diffusion models and therefore mirror biases and (mis-)conceptions that are present
in their training data. Although efforts were made to reduce the inclusion of explicit pornographic material, **we do not recommend using the provided weights for services or products without additional safety mechanisms and considerations.
The weights are research artifacts and should be treated as such.**
Details on the training procedure and data, as well as the intended use of the model can be found in the corresponding [model card](https://huggingface.co/stabilityai/stable-diffusion-2).
The weights are available via [the StabilityAI organization at Hugging Face](https://huggingface.co/StabilityAI) under the [CreativeML Open RAIL++-M License](LICENSE-MODEL). 

## Shout-Outs
- Thanks to [Hugging Face](https://huggingface.co/) and in particular [Apolinário](https://github.com/apolinario)  for support with our model releases!
- Stable Diffusion would not be possible without [LAION](https://laion.ai/) and their efforts to create open, large-scale datasets.
- The [DeepFloyd team](https://twitter.com/deepfloydai) at Stability AI, for creating the subset of [LAION-5B](https://laion.ai/blog/laion-5b/) dataset used to train the model.
- Stable Diffusion 2.0 uses [OpenCLIP](https://laion.ai/blog/large-openclip/), trained by [Romain Beaumont](https://github.com/rom1504).  
- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
Thanks for open-sourcing!
- [CompVis](https://github.com/CompVis/stable-diffusion) initial stable diffusion release
- [Patrick](https://github.com/pesser)'s [implementation](https://github.com/runwayml/stable-diffusion/blob/main/scripts/inpaint_st.py) of the streamlit demo for inpainting.
- `img2img` is an application of [SDEdit](https://arxiv.org/abs/2108.01073) by [Chenlin Meng](https://cs.stanford.edu/~chenlin/) from the [Stanford AI Lab](https://cs.stanford.edu/~ermon/website/). 
- [Kat's implementation]((https://github.com/CompVis/latent-diffusion/pull/51)) of the [PLMS](https://arxiv.org/abs/2202.09778) sampler, and [more](https://github.com/crowsonkb/k-diffusion).
- [DPMSolver](https://arxiv.org/abs/2206.00927) [integration](https://github.com/CompVis/stable-diffusion/pull/440) by [Cheng Lu](https://github.com/LuChengTHU).
- Facebook's [xformers](https://github.com/facebookresearch/xformers) for efficient attention computation.
- [MiDaS](https://github.com/isl-org/MiDaS) for monocular depth estimation.


## License

The code in this repository is released under the MIT License.

The weights are available via [the StabilityAI organization at Hugging Face](https://huggingface.co/StabilityAI), and released under the [CreativeML Open RAIL++-M License](LICENSE-MODEL) License.

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```