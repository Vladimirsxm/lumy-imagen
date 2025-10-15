import os
os.environ.setdefault("HF_HOME", "/opt/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/hf_cache/transformers")
os.environ.setdefault("INSIGHTFACE_HOME", "/opt/insightface")

import io, time, base64, hashlib, requests
from PIL import Image
import torch, runpod
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

try:
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
except Exception:
    StableDiffusionXLRefinerPipeline = None  # type: ignore

try:
    from diffusers import AutoencoderKL  # type: ignore
except Exception:
    AutoencoderKL = None  # type: ignore

# -------- IP-Adapter FaceID (multi-chemins et multi-classes) --------
IPAdapterFaceClass = None  # type: ignore
_ipadapter_import_src = "none"
_ipadapter_class_name = "none"

try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL as _FaceClass  # type: ignore
    IPAdapterFaceClass = _FaceClass
    _ipadapter_import_src = "ip_adapter.ip_adapter_faceid"
    _ipadapter_class_name = "IPAdapterFaceIDPlusXL"
except Exception as e1:
    try:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus as _FaceClass  # type: ignore
        IPAdapterFaceClass = _FaceClass
        _ipadapter_import_src = "ip_adapter.ip_adapter_faceid"
        _ipadapter_class_name = "IPAdapterFaceIDPlus"
    except Exception as e2:
        try:
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID as _FaceClass  # type: ignore
            IPAdapterFaceClass = _FaceClass
            _ipadapter_import_src = "ip_adapter.ip_adapter_faceid"
            _ipadapter_class_name = "IPAdapterFaceID"
        except Exception as e3:
            try:
                from diffusers.pipelines.ip_adapter import IPAdapterFaceIDPlusXL as _FaceClass  # type: ignore
                IPAdapterFaceClass = _FaceClass
                _ipadapter_import_src = "diffusers.pipelines.ip_adapter"
                _ipadapter_class_name = "IPAdapterFaceIDPlusXL"
            except Exception as e4:
                try:
                    from diffusers import IPAdapterFaceIDPlusXL as _FaceClass  # type: ignore
                    IPAdapterFaceClass = _FaceClass
                    _ipadapter_import_src = "diffusers"
                    _ipadapter_class_name = "IPAdapterFaceIDPlusXL"
                except Exception as e5:
                    IPAdapterFaceClass = None
                    print("[handler] IP-Adapter import failed:",
                          repr(e1), repr(e2), repr(e3), repr(e4), repr(e5))

# diag versions
try:
    import diffusers as _df; _df_ver = getattr(_df, "__version__", "?")
except Exception:
    _df_ver = "?"
try:
    import ip_adapter as _ipa  # type: ignore
    _ipa_ver = getattr(_ipa, "__version__", "imported")
except Exception:
    _ipa_ver = "not-imported"
print(f"[diag] diffusers={_df_ver}, ip-adapter={_ipa_ver}, ipadapter_import_src={_ipadapter_import_src}, class={_ipadapter_class_name}")

# -------- Variables globales --------
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
        "has_ipadapter_class": IPAdapterFaceClass is not None,
        "pipeline_kind": CURRENT_PIPELINE_KIND,
        "diffusers": _df_ver,
        "ip-adapter": _ipa_ver,
        "ipadapter_import_src": _ipadapter_import_src,
        "ipadapter_class": _ipadapter_class_name,
    }

    if (
        isinstance(pipeline, StableDiffusionXLPipeline)
        and reference_face_b64
        and IPAdapterFaceClass is not None
    ):
        try:
            from PIL import Image as _PILImage

            ref_str = reference_face_b64
            if ref_str.startswith("data:"):
                ref_str = ref_str.split(",", 1)[1]
            ref_img = _PILImage.open(io.BytesIO(base64.b64decode(ref_str))).convert("RGB")

            global IP_FACEID_ADAPTER
            if IP_FACEID_ADAPTER is None:
                # Télécharger le checkpoint IP-Adapter FaceID
                from huggingface_hub import hf_hub_download
                
                ip_ckpt_repo = "h94/IP-Adapter-FaceID"
                weight_name = os.getenv("IPADAPTER_WEIGHT", "ip-adapter-faceid_sdxl.bin")
                
                try:
                    # Télécharger le poids depuis HF
                    ip_ckpt_path = hf_hub_download(repo_id=ip_ckpt_repo, filename=weight_name)
                    debug["ipadapter_weight_downloaded"] = weight_name
                except Exception as e_download:
                    # Fallback sur un autre poids si le premier échoue
                    try:
                        weight_name = "ip-adapter-faceid-plusv2_sdxl.bin"
                        ip_ckpt_path = hf_hub_download(repo_id=ip_ckpt_repo, filename=weight_name)
                        debug["ipadapter_weight_downloaded"] = weight_name
                    except Exception as e_download2:
                        debug["ipadapter_download_error"] = f"{str(e_download)} | {str(e_download2)}"
                        raise e_download2
                
                # Initialiser avec la signature correcte: (sd_pipe, ip_ckpt_path, device)
                try:
                    IP_FACEID_ADAPTER = IPAdapterFaceClass(pipeline, ip_ckpt_path, "cuda")  # type: ignore
                    
                    # Patch encode_prompt pour SDXL (retourne 6 valeurs au lieu de 2)
                    original_encode_prompt = IP_FACEID_ADAPTER.pipe.encode_prompt
                    def patched_encode_prompt(*args, **kwargs):
                        result = original_encode_prompt(*args, **kwargs)
                        # SDXL retourne (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, ...)
                        # IPAdapterFaceID attend seulement (prompt_embeds, negative_prompt_embeds)
                        if isinstance(result, tuple) and len(result) > 2:
                            return result[0], result[1]
                        return result
                    IP_FACEID_ADAPTER.pipe.encode_prompt = patched_encode_prompt
                    
                    debug["ipadapter_init_variant"] = "sd_pipe_local_path_cuda_patched"
                except Exception as e_init:
                    debug["ipadapter_init_error"] = str(e_init)
                    raise e_init

            cache_key = _hash_b64_image(ref_str)
            faceid_embeds = FACE_EMBED_CACHE.get(cache_key)
            if faceid_embeds is None:
                # IPAdapterFaceID n'a pas de méthode get_face_embeds/get_face_embed
                # Il faut extraire les embeddings manuellement avec InsightFace
                from insightface.app import FaceAnalysis
                import numpy as np
                
                # Forcer CPU provider uniquement pour éviter les problèmes CUDA
                # et utiliser le modèle pré-téléchargé
                app = None
                # InsightFace ignore INSIGHTFACE_HOME, on force /opt/insightface
                insightface_home = "/opt/insightface"
                debug["insightface_home"] = insightface_home
                
                for model_name in ["antelopev2", "buffalo_l"]:
                    model_path = f"{insightface_home}/models/{model_name}"
                    # Vérifier si le modèle existe localement
                    if not os.path.exists(model_path):
                        debug[f"insightface_{model_name}_not_found"] = model_path
                        continue
                    
                    try:
                        # Utiliser allowed_modules=None pour charger depuis le chemin local
                        app = FaceAnalysis(
                            name=model_name,
                            root=insightface_home,
                            providers=['CPUExecutionProvider'],
                            allowed_modules=['detection', 'recognition']
                        )
                        # ctx_id=-1 pour CPU
                        app.prepare(ctx_id=-1, det_size=(640, 640))
                        debug["insightface_model"] = model_name
                        debug["insightface_provider"] = "CPU"
                        break
                    except Exception as e_model:
                        debug[f"insightface_{model_name}_error"] = str(e_model)
                        continue
                
                if app is None:
                    raise RuntimeError(f"Failed to load any InsightFace model from {insightface_home}/models/")
                
                # Convertir PIL en numpy array
                ref_img_np = np.array(ref_img)
                
                # Extraire le visage
                try:
                    faces = app.get(ref_img_np)
                    if len(faces) == 0:
                        raise ValueError("No face detected in reference image")
                    
                    # Prendre le premier visage (ou le plus grand)
                    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                    debug["faces_detected"] = len(faces)
                except Exception as e_face:
                    debug["face_extraction_error"] = str(e_face)
                    raise e_face
                
                FACE_EMBED_CACHE[cache_key] = faceid_embeds

            # Appel generate avec compatibilité multi-signatures
            result_img = None
            try:
                result_img = IP_FACEID_ADAPTER.generate(
                    prompt=final_prompt,
                    negative_prompt=negative,
                    faceid_embeds=faceid_embeds,
                    num_samples=1,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    scale=ip_weight,
                    seed=seed,
                )
            except TypeError as e1:
                debug["generate_attempt1_error"] = str(e1)
                try:
                    result_img = IP_FACEID_ADAPTER.generate(
                        prompt=final_prompt,
                        negative_prompt=negative,
                        faceid_embeds=faceid_embeds,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        ip_adapter_scale=ip_weight,
                        generator=gen,
                    )
                except TypeError as e2:
                    debug["generate_attempt2_error"] = str(e2)
                    result_img = IP_FACEID_ADAPTER.generate(
                        prompt=final_prompt,
                        negative_prompt=negative,
                        faceid_embeds=faceid_embeds,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=gen,
                    )
            
            # Extraire l'image du résultat (peut être PIL.Image, list, ou tuple)
            if isinstance(result_img, list):
                image = result_img[0]
            elif isinstance(result_img, tuple):
                image = result_img[0]
            else:
                image = result_img
            
            used_faceid = True
        except Exception as e:
            import traceback
            print(f"[handler] FaceID fallback: {e}")
            traceback.print_exc()
            image = None
            debug["faceid_error"] = str(e)
            debug["faceid_traceback"] = traceback.format_exc()

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