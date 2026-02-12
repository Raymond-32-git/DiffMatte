import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
import numpy as np

from detectron2.structures import ImageList

class DifMatte(nn.Module):
    def __init__(self,
                 *,
                 model,
                 diffusion,
                 input_format,
                 size_divisibility,
                 args
                 ):
        super(DifMatte, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.args = args


    def forward(self, batched_inputs, t=None, self_align_stage1=False, self_align_stage2=False, time_interval=0, sample_iter=10, alpha=None):   # during inference t is not necessary

        if time_interval != 0:
            sample_list = [iter + time_interval -1 for iter in range(sample_iter+1)]
        else:
            sample_list = None

        if self.training:
            if self.diffusion.uniform_timesteps:
                terms, images, times = self.diffusion.training_losses(self.model, batched_inputs, self_align_stage1 = self_align_stage1, self_align_stage2 = self_align_stage2)          
                return terms, images, times
            else:
                terms, images = self.diffusion.training_losses(self.model, batched_inputs, t)          
                return terms, images
        else:
            # 1. INITIALIZE NOISE: Use Trimap to hint background (-1) and foreground (1)
            device = batched_inputs["trimap"].device
            shape = batched_inputs["trimap"].shape
            noise = torch.randn(*shape, device=device)
            trimap = batched_inputs["trimap"]
            init_mask = trimap.clone() * 2 - 1
            noise = torch.where(trimap == 0.5, noise, init_mask)

            if self.diffusion.uniform_timesteps:
                sample_fn = (
                    self.diffusion.ddpm_sample if not self.args["use_ddim"] else self.diffusion.ddim_sample
                )
                sample = sample_fn(
                    self.model,
                    shape,
                    batched_inputs,
                    noise=noise,
                    sample_list=sample_list,
                    GTalpha=alpha
                )
            else:
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.args["use_ddim"] else self.diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    self.model,
                    shape,
                    batched_inputs,
                    noise=noise,
                    clip_denoised=self.args["clip_denoised"],
                    model_kwargs=None,
                )

            if sample_list == None:
                # Value mapping:
                # UniformGauss (ViTB) already unnormalizes to [0, 1] internally.
                # Standard GaussianDiffusion returns [-1, 1].
                if self.diffusion.uniform_timesteps:
                    alpha_pred = torch.clamp(sample, 0., 1.)
                else:
                    alpha_pred = (sample + 1) / 2
                    alpha_pred = torch.clamp(alpha_pred, 0., 1.)
                
                # Double check background and foreground based on original trimap
                # (trimap range is [0.0, 0.5, 1.0])
                alpha_pred[trimap == 0] = 0.0
                alpha_pred[trimap > 0.9] = 1.0 

                alpha_pred = alpha_pred[0, 0, ...].detach().data.cpu().numpy() * 255
                return alpha_pred
            else:
                return sample