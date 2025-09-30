import os, io, time, base64, requests
import runpod
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

pipeline = None

def init_pipeline():
    global pipeline
    if pipeline is None:
        model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-2-1")
        if "xl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)

def upload_presigned(png_bytes: bytes, presigned_put_url: str):
    r = requests.put(presigned_put_url, data=png_bytes, headers={'Content-Type':'image/png'}, timeout=60)
    r.raise_for_status()
    return presigned_put_url.split('?',1)[0]
 
def image_to_base64(img: Image.Image, fmt: str = "WEBP", quality: int = 90) -> tuple[str, str]:
    """Encode une image en base64 (WEBP par défaut pour réduire la taille).
    Retourne (b64, mime).
    """
    bio = io.BytesIO()
    fmt = fmt.upper()
    mime = 'image/webp' if fmt == 'WEBP' else 'image/png'
    if fmt == 'WEBP':
        img.save(bio, format='WEBP', quality=quality)
    else:
        img.save(bio, format='PNG')
    b64 = base64.b64encode(bio.getvalue()).decode('ascii')
    return b64, mime

def handler(event):
    data = event.get("input", event)
    prompt = data.get("prompt","A children’s book illustration")
    scene = data.get("scene","in a sunny park")
    negative = data.get("negative_prompt","close-up, centered composition, distorted face, huge head, blurry, low quality")
    seed = int(data.get("seed", 42))
    steps = int(data.get("steps", 28))
    width = int(data.get("width", 1024))
    height = int(data.get("height", 768))
    s3_put = data.get("s3_presigned_put")
    return_base64 = bool(data.get("return_base64", True))
    job_id = data.get("job_id", f"job_{int(time.time())}")

    comp_txt = "rule of thirds, subject on the left third, medium distance, not centered, not close-up"
    style = "children's book illustration, soft watercolor, gentle outlines"
    final_prompt = f"{prompt}, {scene}, {comp_txt}, {style}"

    init_pipeline()
    t0 = time.time()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if isinstance(pipeline, StableDiffusionXLPipeline):
        image = pipeline(
            prompt=final_prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            width=width, height=height,
            generator=gen
        ).images[0]
    else:
        image = pipeline(
            prompt=final_prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=7.5,
            width=width, height=height,
            generator=gen
        ).images[0]

    # Retour: soit base64, soit upload S3 si URL fournie
    result = {"status":"ok","job_id":job_id, "elapsed_s": round(time.time()-t0,3)}
    if return_base64 or not s3_put:
        b64, mime = image_to_base64(image, fmt=data.get("format","WEBP"), quality=int(data.get("quality",90)))
        result.update({"image_base64": b64, "mime": mime})
    if s3_put:
        png_buf = io.BytesIO()
        image.save(png_buf, format='PNG')
        out_url = upload_presigned(png_buf.getvalue(), s3_put)
        result.update({"s3_url": out_url})
    return result

runpod.serverless.start({"handler": handler})


