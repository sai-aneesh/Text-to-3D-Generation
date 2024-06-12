import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.utils.ops import perpendicular_component

from peft import LoraConfig,get_peft_model

def print_trainable_parameters(model,model_2):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    for name, module in model_2.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module,nn.Conv2d):# and name in target_modules:
            linear_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    threestudio.info(f"Selected Trainable parameters: {linear_params}")
    threestudio.info(f"Selected Trainable parameters Percentage: {100*linear_params/total_params:.2f}%")
    threestudio.info(f"LORA Trainable parameters: {trainable_params}")
    threestudio.info(f"LORA Aprrox Percentage : {100*trainable_params/linear_params:.2f}%")
    threestudio.info(f"Total parameters: {total_params}")
    threestudio.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

@threestudio.register("mvdream-vsd-guidance")
class MVDream_guidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "sd-v2.1-base-4view" 
        ckpt_path: Optional[str] = None 
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None 
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = False
        recon_std_rescale: float = 0.5


        lora_rank: int = 12
        lora_alpha: int = 12
        lora_dropout: float = 0.1
        lora_lr: float = 0.001

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading MV-Dream Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path)
        self.lora_model = copy.deepcopy(self.model)
        config = LoraConfig(
                        r=self.Config.lora_rank,
                        lora_alpha=self.Config.lora_alpha,
                        target_modules=['proj_out', 'to_k', 'c_proj', 'to_q', 'proj_in', 'to_v', 'c_fc', 'proj','conv_out', 'conv2','conv_in','conv'],
                        lora_dropout=self.Config.lora_dropout,
                        bias="none",
                        )
        
        self.lora_model = get_peft_model(self.lora_model, config)

        print_trainable_parameters(self.lora_model,self.model)
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        self.grad_clip_val: Optional[float] = None
        self.to(self.device)


        self.lora_optim = torch.optim.Adam(self.lora_model.parameters(), lr=self.cfg.lora_lr)

        threestudio.info(f"Loaded MV-Dream VSD Diffusion!")

    def get_camera_cond(self, 
            camera: Float[Tensor, "B 4 4"],
            fovy = None,
    ):

        if self.cfg.camera_condition_type == "rotation": 
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents 

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy = None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)


        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                pred_rgb = F.interpolate(rgb_BCHW, (self.cfg.image_size, self.cfg.image_size), mode='bilinear', align_corners=False)
                latents = self.encode_images(pred_rgb)

        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)


        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        t_expand = t.repeat(text_embeddings.shape[0])
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            latent_model_input = torch.cat([latents_noisy] * 2)
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2,1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # lora Prediction ##TODO: make this better if possible
        noise_lora_pred = self.lora_model.apply_model(latent_model_input, t_expand, context)
        noise_lora_pred_text, noise_lora_pred_uncond = noise_lora_pred.chunk(2)
        noise_lora_pred = noise_lora_pred_uncond + 1.0 * (noise_lora_pred_text - noise_lora_pred_uncond)
        loss_lora = 0.5 * F.mse_loss(noise, noise_lora_pred, reduction="sum") / latents.shape[0]
        loss_lora.backward()
        self.lora_optim.step()
        self.lora_optim.zero_grad()



        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)


        w = (1 - self.model.alphas_cumprod[t])
        with torch.no_grad():
            grad = w * (noise_pred - noise_lora_pred)

        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)