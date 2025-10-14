import os

# Fixer les caches AVANT d'importer les lib lourdes
os.environ.setdefault("HF_HOME", "/opt/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/hf_cache/transformers")
os.environ.setdefault("INSIGHTFACE_HOME", "/opt/insightface")

import io
import time
import base64
import hashlib
import requests
from PIL import Image
import torch
import runpod
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

try:
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
except Exception:
    StableDiffusionXLRefinerPipeline = None  # type: ignore

try:
    from diffusers import AutoencoderKL  # type: ignore
except Exception:
    AutoencoderKL = None  # type: ignore

try:
    # Priorité: package officiel h94/ip-adapter
    from ip_adapter import IPAdapterFaceIDPlusXL  # type: ignore
except Exception:
    try:
        # Fallback si diffusers expose la classe
        from diffusers.pipelines.ip_adapter import IPAdapterFaceIDPlusXL  # type: ignore
    except Exception:
        try:
            from diffusers import IPAdapterFaceIDPlusXL  # type: ignore
        except Exception:
            IPAdapterFaceIDPlusXL = None  # on tracera dans debug


pipeline = None
refiner_pipeline = None
CURRENT_MODEL_ID = None
CURRENT_REFINER_ID = None
CURRENT_PIPELINE_KIND = None  # "sdxl" ou "sd"
IP_FACEID_ADAPTER = None
FACE_EMBED_CACHE: dict[str, object] = {}


def init_pipeline() -> None:
    global pipeline, CURRENT_MODEL_ID, CURRENT_PIPELINE_KIND
    if pipeline is not None:
        return

    model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    CURRENT_MODEL_ID = model_id

    if "xl" in model_id.lower():
        pipeline_local = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        CURRENT_PIPELINE_KIND = "sdxl"
    else:
        pipeline_local = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        CURRENT_PIPELINE_KIND = "sd"

    # VAE optimisé
    try:
        vae_id = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
        if AutoencoderKL is not None and CURRENT_PIPELINE_KIND == "sdxl" and vae_id:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
            pipeline_local.vae = vae.to("cuda")
    except Exception as e:
        print(f"[handler] VAE load skipped: {e}")

    # Optimisations / Scheduler
    try:
        pipeline_local.enable_vae_tiling()
    except Exception:
        pass

    try:
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline_local.scheduler.config)
        if hasattr(scheduler, "use_karras_sigmas"):
            scheduler.use_karras_sigmas = True
        pipeline_local.scheduler = scheduler
    except Exception:
        pass

    pipeline_local.set_progress_bar_config(disable=True)
    pipeline = pipeline_local
    print(f"[handler] Loaded model: {CURRENT_MODEL_ID} (pipeline={CURRENT_PIPELINE_KIND})")


def init_refiner_if_needed() -> None:
    global refiner_pipeline, CURRENT_REFINER_ID
    if refiner_pipeline is not None:
        return
    refiner_id = os.getenv("REFINER_MODEL_ID", "stabilityai/stable-diffusion-xl-refiner-1.0")
    if not refiner_id or StableDiffusionXLRefinerPipeline is None:
        return
    try:
        rp = StableDiffusionXLRefinerPipeline.from_pretrained(refiner_id, torch_dtype=torch.float16).to("cuda")
        try:
            vae_id = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
            if AutoencoderKL is not None and vae_id:
                vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
                rp.vae = vae.to("cuda")
        except Exception:
            pass
        try:
            rp.enable_vae_tiling()
        except Exception:
            pass
        try:
            sched = DPMSolverMultistepScheduler.from_config(rp.scheduler.config)
            if hasattr(sched, "use_karras_sigmas"):
                sched.use_karras_sigmas = True
            rp.scheduler = sched
        except Exception:
            pass
        rp.set_progress_bar_config(disable=True)
        refiner_pipeline = rp
        CURRENT_REFINER_ID = refiner_id
        print(f"[handler] Loaded refiner: {CURRENT_REFINER_ID}")
    except Exception as e:
        print(f"[handler] Refiner load failed: {e}")


def upload_presigned(png_bytes: bytes, presigned_put_url: str) -> str:
    r = requests.put(presigned_put_url, data=png_bytes, headers={"Content-Type": "image/png"}, timeout=60)
    r.raise_for_status()
    return presigned_put_url.split("?", 1)[0]


def image_to_base64(img: Image.Image, fmt: str = "WEBP", quality: int = 90) -> tuple[str, str]:
    bio = io.BytesIO()
    fmt_up = fmt.upper()
    mime = "image/webp" if fmt_up == "WEBP" else "image/png"
    if fmt_up == "WEBP":
        img.save(bio, format="WEBP", quality=quality)
    else:
        img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii"), mime


def _hash_b64_image(b64_str: str) -> str:
    try:
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return hashlib.sha1(raw).hexdigest()
    except Exception:
        return hashlib.sha1(b64_str.encode("utf-8")).hexdigest()


def handler(event):
    data = event.get("input", event)
    prompt = data.get("prompt", "children's book illustration, soft colors")
    scene = data.get("scene", "wide forest at dusk, fireflies, rich readable background")
    negative = data.get(
        "negative_prompt",
        "worst quality, lowres, jpeg artifacts, text, watermark, deformed, extra limbs, close-up, centered, portrait, nsfw",
    )
    seed = int(data.get("seed", 42))
    steps = int(data.get("steps", 30))
    guidance_scale = float(data.get("guidance_scale", 5.5))
    width = int(data.get("width", 1024))
    height = int(data.get("height", 768))
    s3_put = data.get("s3_presigned_put")
    reference_face_b64 = (
        data.get("reference_face_base64")
        or data.get("ip_adapter_face")
        or data.get("ip_adapter_face_base64")
        or data.get("faceid_image")
    )
    if isinstance(reference_face_b64, str):
        reference_face_b64 = reference_face_b64.strip()

    ip_weight = float(data.get("ip_weight", 0.8))
    ip_weight = max(0.0, min(1.2, ip_weight))
    use_refiner = bool(data.get("use_refiner", False))
    refiner_fraction = float(data.get("refiner_fraction", data.get("refiner_strength", 0.8)))
    out_format = data.get("format", "WEBP")
    out_quality = int(data.get("quality", 90))
    return_base64 = bool(data.get("return_base64", True))
    job_id = data.get("job_id", f"job_{int(time.time())}")

    comp_txt = (
        "rule of thirds, medium-wide shot, subject on left or right third, full or 3/4 body,"
        " not centered, not close-up, background detailed but readable"
    )
    style = "children's book illustration, soft watercolor, gentle outlines, smooth shading, pastel colors"
    final_prompt = f"{prompt}, {scene}, {comp_txt}, {style}"

    init_pipeline()
    t0 = time.time()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    used_faceid = False
    used_refiner = False
    image = None

    # SDXL tolère les multiples de 8
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    debug = {
        "has_ref_image": bool(reference_face_b64),
        "has_ipadapter_class": IPAdapterFaceIDPlusXL is not None,
        "pipeline_kind": CURRENT_PIPELINE_KIND,
    }

    if (
        isinstance(pipeline, StableDiffusionXLPipeline)
        and reference_face_b64
        and IPAdapterFaceIDPlusXL is not None
    ):
        try:
            from PIL import Image as _PILImage

            ref_str = reference_face_b64
            if ref_str.startswith("data:"):
                ref_str = ref_str.split(",", 1)[1]
            ref_img = _PILImage.open(io.BytesIO(base64.b64decode(ref_str))).convert("RGB")

            global IP_FACEID_ADAPTER
            if IP_FACEID_ADAPTER is None:
                IP_FACEID_ADAPTER = IPAdapterFaceIDPlusXL(
                    pipeline,
                    "h94/IP-Adapter-FaceID",
                    torch_dtype=torch.float16,
                    device="cuda",
                    weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
                )

            cache_key = _hash_b64_image(ref_str)
            faceid_embeds = FACE_EMBED_CACHE.get(cache_key)
            if faceid_embeds is None:
                try:
                    faceid_embeds = IP_FACEID_ADAPTER.get_face_embeds(ref_img)
                except Exception:
                    faceid_embeds = IP_FACEID_ADAPTER.get_face_embed(ref_img)  # type: ignore
                FACE_EMBED_CACHE[cache_key] = faceid_embeds

            image = IP_FACEID_ADAPTER.generate(
                prompt=final_prompt,
                negative_prompt=negative,
                faceid_embeds=faceid_embeds,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                ip_adapter_scale=ip_weight,
                generator=gen,
            )[0]
            used_faceid = True
        except Exception as e:
            print(f"[handler] FaceID fallback: {e}")
            image = None
            debug["faceid_error"] = str(e)

    if image is None:
        if isinstance(pipeline, StableDiffusionXLPipeline):
            if use_refiner and StableDiffusionXLRefinerPipeline is not None and not used_faceid:
                init_refiner_if_needed()
                if refiner_pipeline is not None:
                    base_out = pipeline(
                        prompt=final_prompt,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        denoising_end=refiner_fraction,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        output_type="latent",
                        generator=gen,
                    )
                    latents = base_out.latents
                    image = refiner_pipeline(
                        prompt=final_prompt,
                        negative_prompt=negative,
                        num_inference_steps=max(10, int(steps * (1.0 - refiner_fraction))),
                        denoising_start=refiner_fraction,
                        guidance_scale=guidance_scale,
                        image=latents,
                        generator=gen,
                    ).images[0]
                    used_refiner = True
                else:
                    image = pipeline(
                        prompt=final_prompt,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=gen,
                    ).images[0]
            else:
                image = pipeline(
                    prompt=final_prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=gen,
                ).images[0]
        else:
            image = pipeline(
                prompt=final_prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
            ).images[0]

    result = {
        "status": "ok",
        "job_id": job_id,
        "elapsed_s": round(time.time() - t0, 3),
        "model_id": CURRENT_MODEL_ID,
        "pipeline": CURRENT_PIPELINE_KIND,
        "used_faceid": used_faceid,
        "used_refiner": used_refiner,
        "ip_weight": ip_weight,
        "debug": debug,
    }

    if return_base64 or not s3_put:
        b64, mime = image_to_base64(image, fmt=out_format, quality=out_quality)
        result.update({"image_base64": b64, "mime": mime})
    if s3_put:
        png_buf = io.BytesIO()
        image.save(png_buf, format="PNG")
        out_url = upload_presigned(png_buf.getvalue(), s3_put)
        result.update({"s3_url": out_url})
    return result


runpod.serverless.start({"handler": handler})
