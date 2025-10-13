import os, io, time, base64, requests, hashlib
import runpod
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
try:
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
except Exception:
    StableDiffusionXLRefinerPipeline = None  # type: ignore
try:
    from diffusers import AutoencoderKL  # type: ignore
except Exception:
    AutoencoderKL = None  # type: ignore
try:
    # Disponible avec diffusers >= 0.29
    from diffusers import IPAdapterFaceIDPlusXL  # type: ignore
except Exception:
    IPAdapterFaceIDPlusXL = None  # type: ignore

pipeline = None
refiner_pipeline = None
CURRENT_MODEL_ID = None
CURRENT_REFINER_ID = None
CURRENT_PIPELINE_KIND = None  # "sdxl" ou "sd"
IP_FACEID_ADAPTER = None  # chargé à la demande
FACE_EMBED_CACHE = {}  # cache simple en mémoire clé -> embeddings

def init_pipeline():
    global pipeline, CURRENT_MODEL_ID, CURRENT_PIPELINE_KIND
    if pipeline is None:
        # Par défaut, bascule sur SDXL base 1.0
        model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        CURRENT_MODEL_ID = model_id
        if "xl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to("cuda")
            CURRENT_PIPELINE_KIND = "sdxl"
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to("cuda")
            CURRENT_PIPELINE_KIND = "sd"
        print(f"[handler] Loaded model: {CURRENT_MODEL_ID} (pipeline={CURRENT_PIPELINE_KIND})")
        # VAE optimisé (notamment pour SDXL)
        try:
            vae_id = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
            if AutoencoderKL is not None and CURRENT_PIPELINE_KIND == "sdxl" and vae_id:
                vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
                pipeline.vae = vae.to("cuda")
                print(f"[handler] Loaded VAE: {vae_id}")
        except Exception as e:
            print(f"[handler] VAE load skipped: {e}")
        # Optimisations mémoire/perf
        try:
            pipeline.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipeline.enable_sequential_cpu_offload()
        except Exception:
            pass
        # Scheduler plus qualitatif/stable que le défaut dans beaucoup de cas
        try:
            scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            # Karras améliore souvent la stabilité/contraste
            if hasattr(scheduler, "use_karras_sigmas"):
                scheduler.use_karras_sigmas = True
            pipeline.scheduler = scheduler
        except Exception:
            pass
        pipeline.set_progress_bar_config(disable=True)


def init_refiner_if_needed():
    global refiner_pipeline, CURRENT_REFINER_ID
    if refiner_pipeline is not None:
        return
    refiner_id = os.getenv("REFINER_MODEL_ID", "stabilityai/stable-diffusion-xl-refiner-1.0")
    if not refiner_id or StableDiffusionXLRefinerPipeline is None:
        return
    try:
        refiner_pipeline = StableDiffusionXLRefinerPipeline.from_pretrained(
            refiner_id, torch_dtype=torch.float16
        ).to("cuda")
        # VAE idem que base si disponible
        try:
            vae_id = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
            if AutoencoderKL is not None and vae_id:
                vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
                refiner_pipeline.vae = vae.to("cuda")
        except Exception:
            pass
        try:
            refiner_pipeline.enable_vae_tiling()
        except Exception:
            pass
        try:
            sched = DPMSolverMultistepScheduler.from_config(refiner_pipeline.scheduler.config)
            if hasattr(sched, "use_karras_sigmas"):
                sched.use_karras_sigmas = True
            refiner_pipeline.scheduler = sched
        except Exception:
            pass
        refiner_pipeline.set_progress_bar_config(disable=True)
        CURRENT_REFINER_ID = refiner_id
        print(f"[handler] Loaded refiner: {CURRENT_REFINER_ID}")
    except Exception as e:
        print(f"[handler] Refiner load failed: {e}")

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


def _hash_b64_image(b64_str: str) -> str:
    try:
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return hashlib.sha1(raw).hexdigest()
    except Exception:
        return hashlib.sha1(b64_str.encode('utf-8')).hexdigest()

def handler(event):
    data = event.get("input", event)
    prompt = data.get("prompt","children's book illustration, soft colors")
    scene = data.get("scene","wide forest at dusk, fireflies, rich readable background")
    negative = data.get(
        "negative_prompt",
        "worst quality, lowres, jpeg artifacts, text, watermark, deformed, extra limbs, close-up, centered, portrait, nsfw"
    )
    seed = int(data.get("seed", 42))
    steps = int(data.get("steps", 30))
    guidance_scale = float(data.get("guidance_scale", 5.5))
    width = int(data.get("width", 1024))
    height = int(data.get("height", 768))
    s3_put = data.get("s3_presigned_put")
    reference_face_b64 = data.get("reference_face_base64")
    ip_weight = float(data.get("ip_weight", 0.7))
    use_refiner = bool(data.get("use_refiner", False))
    refiner_fraction = float(data.get("refiner_fraction", 0.8))  # portion du denoising réalisée par la base
    out_format = data.get("format", "WEBP")
    out_quality = int(data.get("quality", 90))
    return_base64 = bool(data.get("return_base64", True))
    job_id = data.get("job_id", f"job_{int(time.time())}")

    # Composition et style par défaut pensés pour histoires pour enfants
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
    # S'assurer des dimensions multiples de 64
    width = max(64, (width // 64) * 64)
    height = max(64, (height // 64) * 64)
    if (
        isinstance(pipeline, StableDiffusionXLPipeline)
        and reference_face_b64
        and IPAdapterFaceIDPlusXL is not None
    ):
        try:
            import numpy as np  # insightface attend souvent du numpy
            from PIL import Image as _PILImage
            # Décoder l'image de référence
            if reference_face_b64.startswith("data:"):
                reference_face_b64 = reference_face_b64.split(",", 1)[1]
            ref_img = _PILImage.open(io.BytesIO(base64.b64decode(reference_face_b64))).convert("RGB")

            global IP_FACEID_ADAPTER
            if IP_FACEID_ADAPTER is None:
                # Télécharge/charge le modèle FaceID Plus v2 pour SDXL
                IP_FACEID_ADAPTER = IPAdapterFaceIDPlusXL(
                    pipeline,
                    "h94/IP-Adapter-FaceID",
                    torch_dtype=torch.float16,
                    device="cuda",
                    weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
                )

            # Extraire/cacher l'embedding visage
            cache_key = _hash_b64_image(reference_face_b64)
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
            )[0]
            used_faceid = True
        except Exception as e:
            print(f"[handler] FaceID fallback: {e}")
            image = None

    if image is None:
        # Génération standard
        if isinstance(pipeline, StableDiffusionXLPipeline):
            # Si refiner demandé et disponible et pas FaceID (plus simple/robuste)
            if use_refiner and StableDiffusionXLRefinerPipeline is not None and not used_faceid:
                init_refiner_if_needed()
                if refiner_pipeline is not None:
                    # Étape 1: base jusqu'à une fraction (ex: 0.8) en latents
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
                    # Étape 2: refiner à partir de la même fraction
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

    # Retour: soit base64, soit upload S3 si URL fournie
    result = {"status":"ok","job_id":job_id, "elapsed_s": round(time.time()-t0,3),
              "model_id": CURRENT_MODEL_ID, "pipeline": CURRENT_PIPELINE_KIND,
              "used_faceid": used_faceid, "used_refiner": used_refiner,
              "ip_weight": ip_weight}
    if return_base64 or not s3_put:
        b64, mime = image_to_base64(image, fmt=out_format, quality=out_quality)
        result.update({"image_base64": b64, "mime": mime})
    if s3_put:
        png_buf = io.BytesIO()
        image.save(png_buf, format='PNG')
        out_url = upload_presigned(png_buf.getvalue(), s3_put)
        result.update({"s3_url": out_url})
    return result

runpod.serverless.start({"handler": handler})


