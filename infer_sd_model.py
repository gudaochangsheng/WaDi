import torch
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DiffusionPipeline
from safetensors.torch import load_file


device = "cuda"
dtype = torch.float32

base_model = "sd1-5/sd2-1"
rotated_unet_path = "rotated_unet.safetensors"
prompt = "a cat"
output_path = "result.png"


@torch.no_grad()
def main():
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.load_state_dict(load_file(rotated_unet_path, device=device), strict=True)
    unet.eval()

    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()

    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    pipe = DiffusionPipeline.from_pretrained(
        base_model, scheduler=scheduler, torch_dtype=dtype
    )
    pipe.text_encoder = pipe.text_encoder.to(device).eval()

    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    prompt_embeds = pipe.text_encoder(text_inputs.input_ids)[0].to(dtype)

    latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
    t = torch.full((1,), scheduler.config.num_train_timesteps - 1, device=device, dtype=torch.long)

    alpha_T = 0.0047 ** 0.5
    sigma_T = (1 - 0.0047) ** 0.5

    model_pred = unet(latents, t, prompt_embeds).sample
    pred_latents = (latents - sigma_T * model_pred) / alpha_T
    pred_latents = pred_latents / vae.config.scaling_factor

    image = vae.decode(pred_latents).sample[0]
    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()

    Image.fromarray(image).save(output_path)
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()