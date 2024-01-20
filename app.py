import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import argparse
from omegaconf import OmegaConf
import os
from models import get_models
from diffusers.utils.import_utils import is_xformers_available
from tca.tca_transform import tca_transform_model
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from datasets import video_transforms
from torchvision import transforms
from utils import mask_generation_before
from backend import auto_inpainting
from einops import rearrange
import torchvision
import sys
from PIL import Image
from ip_adapter.ip_adapter_transform import ip_scale_set, ip_transform_model
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers.image_transforms import convert_to_rgb
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    # sys.path.append(os.getcwd())
    sys.path.append(os.path.split(sys.path[0])[0])
    # 代码解释
    # sys.path[0]                   : 得到C:\Users\maxu\Desktop\blog_test\pakage2
    # os.path.split(sys.path[0])    : 得到['C:\Users\maxu\Desktop\blog_test',pakage2']
    # mmcls 里面跨包引用是因为安装了mmcls
    
    import utils

    from diffusion import create_diffusion
    from download import find_model
    
    
def auto_inpainting(video_input, masked_video, mask, prompt, image, vae, text_encoder, image_encoder, diffusion, model, device, cfg_scale, img_cfg_scale, negative_prompt=""):
    global use_fp16
    image_prompt_embeds = None
    if prompt is None:
        prompt = ""
    if image is not None:
        clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values
        clip_image_embeds = image_encoder(clip_image.to(device)).image_embeds
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds).to(device)
        image_prompt_embeds = torch.cat([clip_image_embeds, uncond_clip_image_embeds], dim=0)
        image_prompt_embeds = rearrange(image_prompt_embeds, '(b n) c -> b n c', b=2).contiguous()
        model = ip_scale_set(model, img_cfg_scale)
        if use_fp16:
            image_prompt_embeds = image_prompt_embeds.to(dtype=torch.float16)
    b, f, c, h, w = video_input.shape
    latent_h = video_input.shape[-2] // 8
    latent_w = video_input.shape[-1] // 8

    if use_fp16:
        z = torch.randn(1, 4, 16, latent_h, latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, 16, latent_h, latent_w, device=device) # b,c,f,h,w

    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    masked_video = torch.cat([masked_video] * 2)
    mask = torch.cat([mask] * 2)
    z = torch.cat([z] * 2)
    prompt_all = [prompt] + [negative_prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                        class_labels=None, 
                        cfg_scale=cfg_scale,
                        use_fp16=use_fp16,
                        ip_hidden_states=image_prompt_embeds)
    
    # Sample images:
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
        mask=mask, x_start=masked_video, use_concat=True
    )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip


def auto_inpainting_temp_split(video_input, masked_video, mask, prompt, image, vae, text_encoder, image_encoder, diffusion, model, device, scfg_scale, tcfg_scale, img_cfg_scale, negative_prompt=""):
    global use_fp16
    image_prompt_embeds = None
    if prompt is None:
        prompt = ""
    if image is not None:
        clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values
        clip_image_embeds = image_encoder(clip_image.to(device)).image_embeds
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds).to(device)
        image_prompt_embeds = torch.cat([clip_image_embeds, clip_image_embeds, uncond_clip_image_embeds], dim=0)
        image_prompt_embeds = rearrange(image_prompt_embeds, '(b n) c -> b n c', b=3).contiguous()
        model = ip_scale_set(model, img_cfg_scale)
        if use_fp16:
            image_prompt_embeds = image_prompt_embeds.to(dtype=torch.float16)
    b, f, c, h, w = video_input.shape
    latent_h = video_input.shape[-2] // 8
    latent_w = video_input.shape[-1] // 8

    if use_fp16:
        z = torch.randn(1, 4, 16, latent_h, latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, 16, latent_h, latent_w, device=device) # b,c,f,h,w

    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    masked_video = torch.cat([masked_video] * 3)
    mask = torch.cat([mask] * 3)
    z = torch.cat([z] * 3)
    prompt_all = [prompt] + [prompt] + [negative_prompt]
    prompt_temp = [prompt] + [""] + [""]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    temporal_text_prompt = text_encoder(text_prompts=prompt_temp, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                        class_labels=None, 
                        scfg_scale=scfg_scale,
                        tcfg_scale=tcfg_scale,
                        use_fp16=use_fp16,
                        ip_hidden_states=image_prompt_embeds,
                        encoder_temporal_hidden_states=temporal_text_prompt)
    
    # Sample images:
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg_temp_split, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
        mask=mask, x_start=masked_video, use_concat=True
    )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip


# ========================================
#             Model Initialization
# ========================================
device = None
output_path = None
use_fp16 = False
model = None
vae = None
text_encoder = None
image_encoder = None
clip_image_processor = None
def init_model():
    global device
    global output_path
    global use_fp16
    global model
    global diffusion
    global vae
    global text_encoder
    global image_encoder
    global clip_image_processor
    print('Initializing ShowMaker', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample_mask.yaml")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = args.save_img_path
    # Load model:
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8
    args.image_h = args.image_size[0]
    args.image_w = args.image_size[1]
    args.latent_h = latent_h
    args.latent_w = latent_w
    print('loading model')
    model = get_models(True, args).to(device)
    model = tca_transform_model(model).to(device)
    model = ip_transform_model(model).to(device)
    if args.use_compile:
        model = torch.compile(model)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
            print("xformer!")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    print('loading succeed')
    model.eval()  # important!
    pretrained_model_path = args.pretrained_model_path
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    text_encoder = TextEmbedder(tokenizer_path=pretrained_model_path + "tokenizer",
                                encoder_path=pretrained_model_path + "text_encoder").to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(device)
    clip_image_processor = CLIPImageProcessor()
    if args.use_fp16:
        print('Warnning: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        use_fp16 = True
    print('Initialization Finished')
init_model()


# ========================================
#             Video Generation
# ========================================
def video_generation(text, image, scfg_scale, tcfg_scale, img_cfg_scale, diffusion):
    with torch.no_grad():
        print("begin generation", flush=True)
        transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.WebVideo320512((320, 512)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        video_frames = torch.zeros(16, 3, 320, 512, dtype=torch.uint8)
        video_frames = transform_video(video_frames)
        video_input = video_frames.to(device).unsqueeze(0) # b,f,c,h,w
        mask = mask_generation_before("all", video_input.shape, video_input.dtype, device)
        masked_video = video_input * (mask == 0)
        if image is not None:
            print(image.shape, flush=True)
            # image = Image.open(image)
        if scfg_scale == tcfg_scale:
            video_clip = auto_inpainting(video_input, masked_video, mask, text, image, vae, text_encoder, image_encoder, diffusion, model, device, scfg_scale, img_cfg_scale)
        else:
            video_clip = auto_inpainting_temp_split(video_input, masked_video, mask, text, image, vae, text_encoder, image_encoder, diffusion, model, device, scfg_scale, tcfg_scale, img_cfg_scale)
        video_clip = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        video_path = os.path.join(output_path, 'video.mp4')
        torchvision.io.write_video(video_path, video_clip, fps=8)
        return video_path
    
    
# ========================================
#             Video Prediction
# ========================================
def video_prediction(text, image, scfg_scale, tcfg_scale, img_cfg_scale, preframe, diffusion):
    with torch.no_grad():
        print("begin generation", flush=True)
        transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.WebVideo320512((320, 512)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        preframe = torch.as_tensor(convert_to_rgb(preframe)).unsqueeze(0)
        zeros = torch.zeros_like(preframe)
        video_frames = torch.cat([preframe] + [zeros] * 15, dim=0).permute(0, 3, 1, 2)
        H_scale = 320 / video_frames.shape[2]
        W_scale = 512 / video_frames.shape[3]
        scale_ = H_scale
        if W_scale < H_scale:
            scale_ = W_scale
        video_frames = torch.nn.functional.interpolate(video_frames, scale_factor=scale_, mode="bilinear", align_corners=False)
        video_frames = transform_video(video_frames)
        video_input = video_frames.to(device).unsqueeze(0) # b,f,c,h,w
        mask = mask_generation_before("first1", video_input.shape, video_input.dtype, device)
        masked_video = video_input * (mask == 0)
        if image is not None:
            print(image.shape, flush=True)
        if scfg_scale == tcfg_scale:
            video_clip = auto_inpainting(video_input, masked_video, mask, text, image, vae, text_encoder, image_encoder, diffusion, model, device, scfg_scale, img_cfg_scale)
        else:
            video_clip = auto_inpainting_temp_split(video_input, masked_video, mask, text, image, vae, text_encoder, image_encoder, diffusion, model, device, scfg_scale, tcfg_scale, img_cfg_scale)
        video_clip = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        video_path = os.path.join(output_path, 'video.mp4')
        torchvision.io.write_video(video_path, video_clip, fps=8)
        return video_path


# ========================================
#      Judge Generation or Prediction
# ========================================
def gen_or_pre(text_input, image_input, scfg_scale, tcfg_scale, img_cfg_scale, preframe_input, diffusion_step):
    default_step = [25, 40, 50, 100, 125, 200, 250]
    difference = [abs(item - diffusion_step) for item in default_step]
    diffusion_step = default_step[difference.index(min(difference))]
    diffusion = create_diffusion(str(diffusion_step))
    if preframe_input is None:
        return video_generation(text_input, image_input, scfg_scale, tcfg_scale, img_cfg_scale, diffusion)
    else:
        return video_prediction(text_input, image_input, scfg_scale, tcfg_scale, img_cfg_scale, preframe_input, diffusion)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(visible=True) as input_raws:
            with gr.Row():
                with gr.Column(scale=1.0):
                    text_input = gr.Textbox(show_label=True, interactive=True, label="Text prompt").style(container=False)
            with gr.Row():
                with gr.Column(scale=0.5):
                    image_input = gr.Image(show_label=True, interactive=True, label="Reference image").style(container=False)
                with gr.Column(scale=0.5):
                    preframe_input = gr.Image(show_label=True, interactive=True, label="First frame").style(container=False)
            with gr.Row():
                with gr.Column(scale=1.0):
                    scfg_scale = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=8,
                        step=0.1,
                        interactive=True,
                        label="Spatial Text Guidence Scale",
                    )
            with gr.Row():
                with gr.Column(scale=1.0):
                    tcfg_scale = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=6.5,
                        step=0.1,
                        interactive=True,
                        label="Temporal Text Guidence Scale",
                    )
            with gr.Row():
                with gr.Column(scale=1.0):
                    img_cfg_scale = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.3,
                        step=0.005,
                        interactive=True,
                        label="Image Guidence Scale",
                    )
            with gr.Row():
                with gr.Column(scale=1.0):
                    diffusion_step = gr.Slider(
                        minimum=20,
                        maximum=250,
                        value=100,
                        step=1,
                        interactive=True,
                        label="Diffusion Step",
                    )
            with gr.Row():
                with gr.Column(scale=0.5, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.5, min_width=0):
                    clear = gr.Button("🔄Clear️")     
        with gr.Column(scale=0.5, visible=True) as video_upload:
            output_video = gr.Video(interactive=False, include_audio=True, elem_id="输出的视频")#.style(height=360)
            # with gr.Column(elem_id="image", scale=0.5) as img_part:
            #     with gr.Tab("Video", elem_id='video_tab'):
                    
            #     with gr.Tab("Image", elem_id='image_tab'):
            #         up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload").style(height=360)
            # upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
    run.click(gen_or_pre, [text_input, image_input, scfg_scale, tcfg_scale, img_cfg_scale, preframe_input, diffusion_step], [output_video])
    
# demo.launch(share=True, enable_queue=True)

demo.launch(server_name="0.0.0.0", server_port=10034, enable_queue=True)