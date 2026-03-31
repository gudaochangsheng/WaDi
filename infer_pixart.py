import torch
from PIL import Image
from diffusers import DDPMScheduler, PixArtAlphaPipeline, AutoencoderKL, Transformer2DModel
from safetensors.torch import load_file


device = "cuda"
dtype = torch.float16

base_model = "PixArt-alpha/PixArt-XL-2-256x256"
transformer_dir = "PixArt-alpha/PixArt-XL-2-256x256"
rotated_transformer_path = "rotated_transformer.safetensors"
prompt = "a man playing with his dog in front of the house"
output_path = "result.png"
resolution = 256


@torch.no_grad()
def main():
    transformer = Transformer2DModel.from_pretrained(
        transformer_dir, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    transformer.load_state_dict(load_file(rotated_transformer_path, device=device), strict=True)
    transformer.eval()

    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    pipe = PixArtAlphaPipeline.from_pretrained(
        base_model,
        scheduler=scheduler,
        torch_dtype=dtype,
    )
    pipe.text_encoder = pipe.text_encoder.to(device).eval()

    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=120,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = pipe.text_encoder(
        input_ids,
        attention_mask=attention_mask
    )[0]

    latents = torch.randn((1, 4, resolution // 8, resolution // 8), device=device, dtype=dtype)
    t = torch.full((1,), scheduler.config.num_train_timesteps - 1, dtype=torch.long, device=device)

    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if transformer.config.sample_size == 128:
        res = torch.tensor([resolution, resolution], device=device, dtype=dtype).unsqueeze(0)
        ar = torch.tensor([[1.0]], device=device, dtype=dtype)
        added_cond_kwargs = {"resolution": res, "aspect_ratio": ar}

    alpha_T = 0.0047 ** 0.5
    sigma_T = (1 - 0.0047) ** 0.5

    model_pred = transformer(
        latents,
        encoder_hidden_states=prompt_embeds,
        timestep=t,
        encoder_attention_mask=attention_mask,
        added_cond_kwargs=added_cond_kwargs,
    )["sample"].chunk(2, 1)[0]

    pred_latents = (latents - sigma_T * model_pred) / alpha_T
    pred_latents = pred_latents / vae.config.scaling_factor

    image = vae.decode(pred_latents).sample[0]
    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()

    Image.fromarray(image).save(output_path)
    print("saved to", output_path)


if __name__ == "__main__":
    main()