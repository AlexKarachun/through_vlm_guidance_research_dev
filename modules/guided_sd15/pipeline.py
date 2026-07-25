import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from tqdm.auto import tqdm
from .ddpm import DDPMSampler

from omegaconf import DictConfig
from hydra.utils import instantiate
from pathlib import Path
import json

from . import loss
from ..utils import save_0_255_tensor_as_img, save_0_255_guidance_diffs, guidance_tag, denoise_tag



WIDTH = 512
HEIGHT = 512


LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
             logging_save_intermediate_finals_path: Path,
             logging_save_guidance_diffs_path: Path,
             logging_save_latents_path: Path,
             logging_save_nablas_path: Path,
             logging_save_general_path: Path,
             
             vlm_criterion: torch.nn.Module,
             
             prompt: str,
             uncond_prompt: str,
             input_image=None,
             strength=0.8, 
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models=None,
             seed=67, 
             device="mps", 
             idle_device=None, 
             tokenizer=None,
             
             progress_position=0,
             progress_leave=False,
             
             guidance_steps=None, 
             cfg: DictConfig=None,
            ):
    
    
    with torch.no_grad():
        
        if not (0 < strength <= 1):
            raise ValueError(f"strength must be in (0, 1], given {strength}")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        
        clip = models['clip']
        clip.to(device)

        def tokenize(text):
            tokens = tokenizer(
                [text],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].to(device)

        if do_cfg:
            cond_tokens = tokenize(prompt)  # B Seq
            cond_context = clip(cond_tokens)  # B Seq Dim
            
            uncond_tokens = tokenize(uncond_prompt)  # B Seq
            uncond_context = clip(uncond_tokens)  # B Seq Dim
            
            context = torch.cat([cond_context, uncond_context]) # 2 Seq Dim
        
        else:
            cond_tokens = tokenize(prompt)  # B Seq
            context = clip(cond_tokens)  # B Seq Dim
        
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f'Unknown sampler {sampler_name}')
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            encoder = models['encoder']
            encoder.to(device)
            
            input_image_tensor = input_image.resize([WIDTH, HEIGHT])
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)  # B H W C
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)  # B C H W 
            input_image_tensor = input_image_tensor.to(device)
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
            
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
            
        diffusion = models['diffusion']
        diffusion.to(device)
        diffusion.requires_grad_(False)
        
    def run_diffusion(model_input, context, time_embedding):
        return diffusion(model_input, context, time_embedding)
        
    
    guidance_steps_loader = tqdm(
        guidance_steps + [None], 
        desc='generation-steps', 
        position=progress_position, 
        leave=progress_leave
    )
    
    skip_first_denoise_steps = 0
    
    
    if cfg.pipeline.logging.save_guidance_diffs:
        intermediate_finals = []
        intermediate_finals_tags = []
    
    if cfg.pipeline.logging.save_yes_no_distributions:
        yes_no_distributions = {}
    
    for g_id, guiding_timestep in enumerate(guidance_steps_loader):
        g_tag = guidance_tag(g_id=g_id, guiding_step=guiding_timestep)
        diffusion.to(device)
        
        timesteps = tqdm(
            sampler.timesteps[skip_first_denoise_steps:],
            total=len(sampler.timesteps[skip_first_denoise_steps:]),
            desc=f'sample-steps {g_tag}', 
            position=progress_position+1, 
            leave=False
        )
        
        for t_id, timestep in enumerate(timesteps, skip_first_denoise_steps):
            timestep = timestep.item()
            d_tag = denoise_tag(t_id=t_id, timestep=timestep)

            # 1 320
            with torch.no_grad():
                time_embedding = get_time_embedding(timestep).to(device)
            
            
            if cfg.pipeline.logging.save_latents:
                latents_path = logging_save_latents_path / f'{g_tag}-{d_tag}-latent.pt'
                torch.save(latents.detach().clone().to(device='cpu'), latents_path)


            # B 4 latent_H latent_W
            step_latent = latents
            
            
            if t_id == guiding_timestep:
                step_latent = step_latent.detach().clone().requires_grad_(True)
                key_step_latent = step_latent
                optimizer = instantiate(cfg.pipeline.optimizer, params=[key_step_latent])
                skip_first_denoise_steps = t_id
                
            model_input = step_latent
            
            if do_cfg:
                # 2*B 4 latent_H latent_W
                model_input = model_input.repeat(2, 1, 1, 1)
            
            
            if cfg.pipeline.guidance.do_grad_checkpointing and model_input.requires_grad:
                model_output = checkpoint(
                    run_diffusion,
                    model_input,
                    context,
                    time_embedding,
                    use_reentrant=False, 
                    preserve_rng_state=False,
                )  # run unet
            else:
                model_output = diffusion(model_input, context, time_embedding)  # run unet
            
            if do_cfg:
                output_cond, output_uncod = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncod) + output_uncod
            
            latents = sampler.step(timestep, step_latent, model_output, try_use_cached_noise=True)
            
                    
        
        decoder = models['decoder']
        decoder.to(device)
        decoder.requires_grad_(False)
        images = decoder(latents)
        
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        
        if cfg.pipeline.logging.save_guidance_diffs:
            intermediate_finals.append(images.detach().clone().to(device='cpu'))
            intermediate_finals_tags.append(g_tag)
        
        if cfg.pipeline.logging.save_intermediate_finals:
            images_copy = images.detach().clone().to(device='cpu')
            
            intermediate_final_path = logging_save_intermediate_finals_path / f'{g_tag}-image_before_update.png'
            save_0_255_tensor_as_img(img_tensor=images_copy, path=intermediate_final_path)
        
        
        images = images.permute(0, 2, 3, 1)  # B C H W -> B H W C

        if guiding_timestep is not None:
            loss, yes_no_distribution = vlm_criterion(
                image_rgb=images[0],
                img_prompt=prompt,
                target_yes=cfg.pipeline.guidance.target_yes,
                give_me_distribution=cfg.pipeline.logging.save_yes_no_distributions
            )
            loss.backward()
            optimizer.step()
            
            if cfg.pipeline.logging.save_yes_no_distributions:
                yes_no_distributions[f'{g_tag}-before_update'] = yes_no_distribution
            
            latents = key_step_latent.detach().clone()


        to_idle(diffusion)
        to_idle(decoder)

        if guiding_timestep is not None and cfg.pipeline.logging.save_nablas:
            nablas_path = logging_save_nablas_path / f'{g_tag}-nabla.pt'
            torch.save(key_step_latent.grad.detach().to(device='cpu'), nablas_path)
        
        if guiding_timestep is not None:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            del loss
            del key_step_latent
            del images
        
        
            
    to_idle(diffusion)
    
    decoder = models['decoder']
    decoder.to(device)
    
    images = decoder(latents)
    to_idle(decoder)
    
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    # B C H W -> B H W C
    images = images.permute(0, 2, 3, 1)
    
    if cfg.pipeline.logging.save_yes_no_distributions:
        _, yes_no_distribution = vlm_criterion(
            image_rgb=images[0],
            img_prompt=prompt,
            target_yes=cfg.pipeline.guidance.target_yes,
            give_me_distribution=cfg.pipeline.logging.save_yes_no_distributions
        )

        
        yes_no_distributions[f'{g_tag}-after_update'] = yes_no_distribution
    
    images = images.to('cpu', torch.uint8).numpy()
    
    
    if cfg.pipeline.logging.save_guidance_diffs:
        save_0_255_guidance_diffs(img_tensors=intermediate_finals, path=logging_save_guidance_diffs_path, tags=intermediate_finals_tags)
    
    if cfg.pipeline.logging.save_yes_no_distributions:    
        with open(logging_save_general_path / 'yes_no_distributions.json', 'w', encoding='utf-8') as f:
            json.dump(yes_no_distributions, f, ensure_ascii=False, indent=4)
    
    return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    x = x - old_min
    x = x * (new_max - new_min) / (old_max - old_min)
    x = x + new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
        
    return x
        

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)  # 160
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]  # 1 160
    
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)





