# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import sys
import math
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion
    from download import find_model

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torchvision import transforms
sys.path.append("..")
from datasets import video_transforms
from decord import VideoReader
from utils import mask_generation_before
from natsort import natsorted
from diffusers.utils.import_utils import is_xformers_available
from tca.tca_transform import tca_transform_model


def get_input(args):
    input_path = args.input_path
    transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.ResizeVideo((args.image_h, args.image_w)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    temporal_sample_func = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval)
    if input_path is not None:
        print(f'loading video from {input_path}')
        if os.path.isdir(input_path):
            file_list = os.listdir(input_path)
            video_frames = []
            if args.mask_type.startswith('onelast'):
                num = int(args.mask_type.split('onelast')[-1])
                # get first and last frame
                first_frame_path = os.path.join(input_path, natsorted(file_list)[0])
                last_frame_path = os.path.join(input_path, natsorted(file_list)[-1])
                first_frame = torch.as_tensor(np.array(Image.open(first_frame_path).convert("RGB"), dtype=np.uint8, copy=True)).unsqueeze(0)
                last_frame = torch.as_tensor(np.array(Image.open(last_frame_path).convert("RGB"), dtype=np.uint8, copy=True)).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                # add zeros to frames
                num_zeros = args.num_frames-2*num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                for i in range(num):
                    video_frames.append(last_frame)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
                video_frames = transform_video(video_frames)
            elif args.mask_type.startswith('video_onelast'):
                num = int(args.mask_type.split('onelast')[-1])
                first_frame_path = os.path.join(input_path, natsorted(file_list)[0])
                last_frame_path = os.path.join(input_path, natsorted(file_list)[-1])
                video_reader_first = VideoReader(first_frame_path)
                video_reader_last = VideoReader(last_frame_path)
                total_frames_first = len(video_reader_first)
                total_frames_last = len(video_reader_last)
                start_frame_ind_f, end_frame_ind_f = temporal_sample_func(total_frames_first)
                start_frame_ind_l, end_frame_ind_l = temporal_sample_func(total_frames_last)
                frame_indice_f = np.linspace(start_frame_ind_f, end_frame_ind_f-1, args.num_frames, dtype=int)
                frame_indice_l = np.linspace(start_frame_ind_l, end_frame_ind_l-1, args.num_frames, dtype=int)
                video_frames_first = torch.from_numpy(video_reader_first.get_batch(frame_indice_f).asnumpy()).permute(0, 3, 1, 2).contiguous()
                video_frames_last = torch.from_numpy(video_reader_last.get_batch(frame_indice_l).asnumpy()).permute(0, 3, 1, 2).contiguous()
                video_frames_first = transform_video(video_frames_first) # f,c,h,w
                video_frames_last = transform_video(video_frames_last)
                num_zeros = args.num_frames-2*num
                video_frames.append(video_frames_first[-num:])
                for i in range(num_zeros):
                    zeros = torch.zeros_like(video_frames_first[0]).unsqueeze(0)
                    video_frames.append(zeros)
                video_frames.append(video_frames_last[:num])
                video_frames = torch.cat(video_frames, dim=0)
                # video_frames = transform_video(video_frames)
                n = num
            else:
                for file in file_list:
                    if file.endswith('jpg') or file.endswith('png'):
                        image = torch.as_tensor(np.array(Image.open(file), dtype=np.uint8, copy=True)).unsqueeze(0)
                        video_frames.append(image)
                    else:
                        continue
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
                video_frames = transform_video(video_frames)
            return video_frames, n
        elif os.path.isfile(input_path):
            _, full_file_name = os.path.split(input_path)
            file_name, extention = os.path.splitext(full_file_name)
            if extention == '.jpg' or extention == '.png':
                # raise TypeError('a single image is not supported yet!!')
                print("reading video from a image")
                video_frames = []
                num = int(args.mask_type.split('first')[-1])
                first_frame = torch.as_tensor(np.array(Image.open(input_path).convert("RGB"), dtype=np.uint8, copy=True)).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                num_zeros = args.num_frames - num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
                H_scale = args.image_h / video_frames.shape[2]
                W_scale = args.image_w / video_frames.shape[3]
                scale_ = H_scale
                if W_scale < H_scale:
                    scale_ = W_scale
                video_frames = torch.nn.functional.interpolate(video_frames, scale_factor=scale_, mode="bilinear", align_corners=False)
                video_frames = transform_video(video_frames)
                return video_frames, n
            elif extention == '.mp4':
                video_reader = VideoReader(input_path)
                total_frames = len(video_reader)
                start_frame_ind, end_frame_ind = temporal_sample_func(total_frames)
                frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, args.num_frames, dtype=int)
                video_frames = torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
                video_frames = transform_video(video_frames)
                n = args.researve_frame
                del video_reader
                return video_frames, n
            else:
                raise TypeError(f'{extention} is not supported !!')
        else:
            raise ValueError('Please check your path input!!')
    else:
        # raise ValueError('Need to give a video or some images')
        print('given video is None, using text to video')
        video_frames = torch.zeros(16,3,args.latent_h,args.latent_w,dtype=torch.uint8)
        args.mask_type = 'all'
        video_frames = transform_video(video_frames)
        n = 0
        return video_frames, n

def auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,):
    # masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    # masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    # masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    # mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_size, latent_size)).unsqueeze(1)
    b,f,c,h,w=video_input.shape
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8

    # prepare inputs
    # video_input = rearrange(video_input, 'b f c h w -> (b f) c h w').contiguous()
    # video_input = vae.encode(video_input).latent_dist.sample().mul_(0.18215)
    # video_input = rearrange(video_input, '(b f) c h w -> b c f h w', b=b).contiguous()
    if args.use_fp16:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device) # b,c,f,h,w


    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    

    # classifier_free_guidance
    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        z = torch.cat([z] * 2)
        prompt_all = [prompt] + [args.negative_prompt]
    else:
        masked_video = masked_video
        mask = mask
        z = z
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                            class_labels=None, 
                            cfg_scale=args.cfg_scale,
                            use_fp16=args.use_fp16,) # tav unet

    # Sample images:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip

def auto_inpainting_temp_split(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,):
    b,f,c,h,w=video_input.shape
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8

    if args.use_fp16:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device) # b,c,f,h,w


    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    
    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 3)
        mask = torch.cat([mask] * 3)
        z = torch.cat([z] * 3)
        prompt_all = [prompt] + [prompt] + [args.negative_prompt]
        prompt_temp = [prompt] + [""] + [""]
    else:
        masked_video = masked_video
        mask = mask
        z = z
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    temporal_text_prompt = text_encoder(text_prompts=prompt_temp, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                            class_labels=None, 
                            cfg_scale=args.cfg_scale,
                            use_fp16=args.use_fp16,
                            encoder_temporal_hidden_states=temporal_text_prompt) # tav unet

    # Sample images:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg_temp_split, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg_temp_split, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip

def main(args):
    # torch.cuda.empty_cache()
    print("--------------------------begin running--------------------------", flush=True)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Setup PyTorch:
    if args.seed:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8
    args.image_h = args.image_size[0]
    args.image_w = args.image_size[1]
    args.latent_h = latent_h
    args.latent_w = latent_w
    print('loading model')
    model = get_models(args.use_mask, args).to(device)
    model = tca_transform_model(model).to(device)
    # model = temp_scale_set(model, 0.98)

    if args.use_compile:
        model = torch.compile(model)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    print('loading succeed')

    model.eval()  # important!
    pretrained_model_path = args.pretrained_model_path
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    text_encoder = TextEmbedder(tokenizer_path=pretrained_model_path + "tokenizer",
                                encoder_path=pretrained_model_path + "text_encoder").to(device)
    if args.use_fp16:
        print('Warnning: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)

    # Labels to condition the model with (feel free to change):
    prompts = args.text_prompt
    class_name = [p + args.additional_prompt for p in prompts]

    if args.use_autoregressive:
        if not os.path.exists(os.path.join(args.save_img_path)):
            os.makedirs(os.path.join(args.save_img_path))
        video_input, researve_frames = get_input(args) # f,c,h,w
        video_input = video_input.to(device).unsqueeze(0) # b,f,c,h,w
        mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device) # b,f,c,h,w
        # TODO: change the first3 to last3
        if args.mask_type.startswith('first') and researve_frames != 0:
            masked_video = torch.cat([video_input[:,-researve_frames:], video_input[:,:-researve_frames]], dim=1) * (mask == 0)
        else:
            masked_video = video_input * (mask == 0)

        all_video = []
        if researve_frames != 0:
            all_video.append(video_input)
        for idx, prompt in enumerate(class_name):
            if idx == 0:
                video_clip = auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,)
                video_clip_ = video_clip.unsqueeze(0)
                all_video.append(video_clip_[:, researve_frames:])
            else:
                researve_frames = args.researve_frame
                if args.mask_type.startswith('first') and researve_frames != 0:
                    masked_video = torch.cat([video_clip_[:,-researve_frames:], video_clip_[:,:-researve_frames]], dim=1) * (mask == 0)
                else:
                    masked_video = video_input * (mask == 0)
                video_clip = auto_inpainting(args, video_clip.unsqueeze(0), masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,)
                video_clip_ = video_clip.unsqueeze(0)
                all_video.append(video_clip_[:, researve_frames:])
            video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
            if args.mask_type.startswith('video_onelast'):
                torchvision.io.write_video(os.path.join(args.save_img_path, 'clip_video_' + '%04d' % idx + '.mp4'), video_[researve_frames:-researve_frames], fps=8)
            else:
                torchvision.io.write_video(os.path.join(args.save_img_path, 'clip_video_' + '%04d' % idx + '.mp4'), video_, fps=8)
        if args.mask_type.startswith('first') and researve_frames != 0:
            all_video = torch.cat(all_video, dim=1).squeeze(0)
            video_ = ((all_video * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
            torchvision.io.write_video(os.path.join(args.save_img_path, 'complete_video' + '.mp4'), video_, fps=8)
        else:
            # all_video = torch.cat(all_video, dim=-1).squeeze(0)
            pass
        print(f'save in {args.save_img_path}')
        return os.path.join(args.save_img_path, 'clip_video_' + '%04d' % idx + '.mp4')


def call_main(input):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample_mask.yaml")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.text_prompt = [input]
    return main(omega_conf)
