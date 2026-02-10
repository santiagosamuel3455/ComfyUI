import os
import sys
import gc
import torch
import time
import psutil
import asyncio
import nest_asyncio
import gradio as gr
import argparse
from types import ModuleType

# --- 0. CONFIGURACIÓN DE ARGUMENTOS (CLI) ---
parser = argparse.ArgumentParser(description="LTX2 Studio Launch Config")

# 1. MODELOS PRINCIPALES (Según tu captura)
parser.add_argument("--unet", type=str, default="ltx-2-19b-distilled-Q3_K_M.gguf", help="Modelo UNET (Q3, Q4, etc)")
parser.add_argument("--vae", type=str, default="LTX2_video_vae_bf16.safetensors", help="Video VAE")
parser.add_argument("--vae_audio", type=str, default="LTX2_audio_vae_bf16.safetensors", help="Audio VAE")
parser.add_argument("--clip1", type=str, default="gemma_3_12B_it_fp4_mixed.safetensors", help="Text Encoder 1")
parser.add_argument("--clip2", type=str, default="ltx-2-19b-embeddings_connector_distill_bf16.safetensors", help="Text Encoder 2")
parser.add_argument("--upscaler", type=str, default="ltx-2-spatial-upscaler-x2-1.0.safetensors", help="Modelo de Upscale (Opcional)")

# 2. LORAS
parser.add_argument("--lora1", type=str, default="None", help="Nombre del LoRA 1")
parser.add_argument("--lora1_st", type=float, default=1.0, help="Fuerza del LoRA 1")
parser.add_argument("--lora2", type=str, default="None", help="Nombre del LoRA 2")
parser.add_argument("--lora2_st", type=float, default=1.0, help="Fuerza del LoRA 2")
parser.add_argument("--lora3", type=str, default="None", help="Nombre del LoRA 3")
parser.add_argument("--lora3_st", type=float, default=1.0, help="Fuerza del LoRA 3")

# 3. GRADIO
parser.add_argument("--share", action="store_true", help="Crear enlace público")
parser.add_argument("--port", type=int, default=7860, help="Puerto de Gradio")

args, _ = parser.parse_known_args()

# --- 1. CONFIGURACIÓN DE RUTAS ---
comfy_path = os.path.abspath('/content/ComfyUI')
os.chdir(comfy_path)

if comfy_path not in sys.path: sys.path.insert(0, comfy_path)
kj_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI_KJNodes')
if kj_path not in sys.path: sys.path.insert(0, kj_path)

nest_asyncio.apply()

# --- 2. PARCHE "DUMMY SERVER" ---
if "server" not in sys.modules:
    dummy_server = ModuleType("server")
    class DummyPromptServer:
        instance = None
        def __init__(self): self.client_id = "headless_script"
        def send_sync(self, event, data, sid=None): pass
    
    server_instance = DummyPromptServer()
    DummyPromptServer.instance = server_instance
    dummy_server.PromptServer = DummyPromptServer
    sys.modules["server"] = dummy_server

# --- 3. IMPORTAR NODOS ---
from nodes import NODE_CLASS_MAPPINGS, init_builtin_extra_nodes, init_external_custom_nodes

try: import pynvml; pynvml.nvmlInit()
except: pass

async def setup_nodes():
    await init_builtin_extra_nodes()
    await init_external_custom_nodes()
    # Mapeos de seguridad
    if "VAELoaderKJ" not in NODE_CLASS_MAPPINGS:
        try:
            import nodes_vae_kj
            NODE_CLASS_MAPPINGS["VAELoaderKJ"] = nodes_vae_kj.VAELoaderKJ
            NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"] = nodes_vae_kj.LTXVAudioVAEDecode
        except: pass

print("⏳ Cargando nodos...")
asyncio.run(setup_nodes())
print("✅ Nodos listos.")

# --- 4. UTILIDADES ---
def get_files(folder, ext=(".safetensors", ".gguf", ".bin", ".pth")):
    path = os.path.join(comfy_path, "models", folder)
    files = []
    if os.path.exists(path):
        for root, _, f_names in os.walk(path):
            for f in f_names:
                if f.endswith(ext):
                    files.append(os.path.relpath(os.path.join(root, f), path))
    return sorted(files)

# --- 5. LIMPIEZA DE MEMORIA ---
def clear_all():
    gc.collect()
    torch.cuda.empty_cache()
    try: torch.cuda.ipc_collect()
    except: pass

def get_v(obj, idx=0):
    if obj is None: return None
    try: return obj[idx] if not hasattr(obj, "result") else obj.result[idx]
    except: return obj[idx]

def get_sigmas(steps):
    return ",".join([str(round(1.0 - (i / steps), 6)) for i in range(steps + 1)])

# --- 6. PROCESO DE GENERACIÓN ---
def generate_process(
    prompt, image, img_strength, seed, steps, res_key, seconds, fps, 
    l1, l1_s, l2, l2_s, l3, l3_s,
    unet_name, vae_name, vae_audio_name, clip1_name, clip2_name
):
    clear_all()
    logs = []
    start_time = time.time()
    
    def log(t):
        print(t)
        logs.append(t)
        return "\n".join(logs)

    RES = {
        "832x480 (16:9)": (832, 480), "1024x1024 (1:1)": (1024, 1024),
        "1280x720 (HD)": (1280, 720), "480x832 (TikTok)": (480, 832)
    }
    w, h = RES[res_key]
    frames = int(seconds * fps) + 1
    
    try:
        with torch.inference_mode():
            # PASO 1: CLIP
            yield log(f"🧠 Cargando CLIPs:\n- {clip1_name}\n- {clip2_name}"), None, None
            clip_data = NODE_CLASS_MAPPINGS["DualCLIPLoader"]().load_clip(
                clip_name1=clip1_name, clip_name2=clip2_name, type="ltxv", device="cpu"
            )
            pos = NODE_CLASS_MAPPINGS["CLIPTextEncode"]().encode(text=prompt, clip=get_v(clip_data, 0))
            neg = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]().zero_out(conditioning=get_v(pos, 0))
            cond = NODE_CLASS_MAPPINGS["LTXVConditioning"]().EXECUTE_NORMALIZED(frame_rate=fps, positive=get_v(pos, 0), negative=get_v(neg, 0))
            cond_pos, cond_neg = get_v(cond, 0), get_v(cond, 1)
            del clip_data, pos, neg, cond; clear_all()
            yield log("🗑️ CLIP liberado."), None, None

            # PASO 2: VAE (VIDEO)
            yield log(f"🖼️ Cargando VAE Video: {vae_name}"), None, None
            vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)
            vae_obj = get_v(vae_loader, 0)

            if image and os.path.exists(image):
                img_d = NODE_CLASS_MAPPINGS["LoadImage"]().load_image(image=image)
                img_r = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]().EXECUTE_NORMALIZED(
                    input=get_v(img_d, 0), scale_method="lanczos", 
                    resize_type={"resize_type": "scale dimensions", "width": w, "height": h, "crop": "center"}
                )
                img_in = get_v(img_r, 0); st, by = img_strength, False
            else:
                img_in = torch.full((1, h, w, 3), 0.5); st, by = 0.0, True

            img_p = NODE_CLASS_MAPPINGS["LTXVPreprocess"]().EXECUTE_NORMALIZED(img_compression=33, image=img_in)
            lat_e = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]().EXECUTE_NORMALIZED(width=w, height=h, length=frames, batch_size=1)
            lat_v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]().EXECUTE_NORMALIZED(
                strength=st, bypass=by, vae=vae_obj, image=get_v(img_p, 0), latent=get_v(lat_e, 0)
            )

            # PASO 2.1: VAE (AUDIO) - ¡Ahora usa el argumento!
            vae_aud = NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(vae_name=vae_audio_name, device="cpu", weight_dtype="fp16")[0]
            lat_a = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]().EXECUTE_NORMALIZED(frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_aud)[0]
            av_in = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]().EXECUTE_NORMALIZED(video_latent=get_v(lat_v, 0), audio_latent=lat_a)[0]

            del vae_loader, vae_obj, img_p, lat_e, lat_v, vae_aud, lat_a; clear_all()
            yield log("🗑️ VAEs liberados."), None, None

            # PASO 3: UNET + LORAS
            yield log(f"⚡ Cargando UNET: {unet_name}"), None, None
            unet = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(unet_name=unet_name)[0]
            
            LoraNode = NODE_CLASS_MAPPINGS.get("LoraLoaderModelOnly", NODE_CLASS_MAPPINGS.get("LoraLoader"))
            
            if l1 and l1 != "None":
                yield log(f"💊 LoRA 1: {l1}"), None, None
                unet = LoraNode().load_lora_model_only(unet, l1, l1_s)[0]
            if l2 and l2 != "None":
                yield log(f"💊 LoRA 2: {l2}"), None, None
                unet = LoraNode().load_lora_model_only(unet, l2, l2_s)[0]
            if l3 and l3 != "None":
                yield log(f"💊 LoRA 3: {l3}"), None, None
                unet = LoraNode().load_lora_model_only(unet, l3, l3_s)[0]

            guider = NODE_CLASS_MAPPINGS["CFGGuider"]().EXECUTE_NORMALIZED(cfg=1, model=unet, positive=cond_pos, negative=cond_neg)
            sigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]().EXECUTE_NORMALIZED(sigmas=get_sigmas(steps))
            
            yield log("🎨 Generando..."), None, None
            sampled = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]().EXECUTE_NORMALIZED(
                noise=NODE_CLASS_MAPPINGS["RandomNoise"]().EXECUTE_NORMALIZED(noise_seed=seed)[0], 
                guider=get_v(guider, 0), sampler=NODE_CLASS_MAPPINGS["KSamplerSelect"]().EXECUTE_NORMALIZED("euler")[0], 
                sigmas=get_v(sigmas, 0), latent_image=get_v(av_in, 0)
            )[0]

            del unet, guider, sigmas, av_in, cond_pos, cond_neg; clear_all()
            yield log("🗑️ UNET liberado."), None, None

            # PASO 4: DECODE
            yield log("🎞️ Decodificando..."), None, None
            vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)
            vae_obj = get_v(vae_loader, 0)

            sep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]().EXECUTE_NORMALIZED(av_latent=get_v(sampled, 0))
            decoded = NODE_CLASS_MAPPINGS["VAEDecode"]().decode(samples=get_v(sep, 0), vae=vae_obj)
            
            # Recargar Audio VAE en GPU para decode rápido
            vae_aud_gpu = NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(vae_name=vae_audio_name, device="cuda", weight_dtype="fp16")[0]
            dec_aud = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]().EXECUTE_NORMALIZED(samples=get_v(sep, 1), audio_vae=vae_aud_gpu)[0]
            
            vid = NODE_CLASS_MAPPINGS["CreateVideo"]().EXECUTE_NORMALIZED(fps=fps, images=get_v(decoded, 0), audio=get_v(dec_aud, 0))
            
            os.makedirs("/content/ComfyUI/output", exist_ok=True)
            out = f"/content/ComfyUI/output/LTX_{seed}_{int(time.time())}.mp4"
            get_v(vid, 0).save_to(out, format="mp4")

            del sep, decoded, dec_aud, vid, vae_loader, vae_obj; clear_all()
            yield log(f"✅ FINALIZADO ({int(time.time()-start_time)}s)"), out, out

    except Exception as e:
        clear_all(); yield log(f"❌ ERROR: {e}"), None, None

# --- 7. MONITOR ---
def stats():
    while True:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0); i = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram = (i.used / i.total) * 100; vram_t = f"{i.used / (1024**3):.1f}GB"
        except: vram, vram_t = 0, "N/A"
        
        c = lambda p: "#22c55e" if p < 60 else "#eab308" if p < 85 else "#ef4444"
        card = "flex:1;background:#1f2937;padding:10px;border-radius:8px;min-width:150px;"
        yield f"""<div style="display:flex;gap:10px;"><div style="{card}"><span style="color:#ddd">⚡ CPU: {cpu}%</span><div style="height:5px;background:#444;margin-top:5px;"><div style="width:{cpu}%;background:{c(cpu)};height:100%"></div></div></div><div style="{card}"><span style="color:#ddd">💾 RAM: {ram}%</span><div style="height:5px;background:#444;margin-top:5px;"><div style="width:{ram}%;background:{c(ram)};height:100%"></div></div></div><div style="{card}"><span style="color:#ddd">📟 VRAM: {vram_t}</span><div style="height:5px;background:#444;margin-top:5px;"><div style="width:{vram}%;background:{c(vram)};height:100%"></div></div></div></div>"""
        time.sleep(1)

# --- 8. UI GRADIO ---
with gr.Blocks(title="LTX2 APP", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.HTML("<h3 style='text-align:center'>🎬 LTX2 STUDIO</h3>")
    monitor = gr.HTML(label="Stats")
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                btn_mon = gr.Button("📊 MONITOR", size="sm")
                btn_clr = gr.Button("🧹 LIMPIAR", variant="stop", size="sm")
                btn_ref = gr.Button("🔄 REFRESCAR", size="sm")

            prompt = gr.Textbox(label="Prompt", value="Cinematic...", lines=2)
            img = gr.Image(type="filepath", label="Imagen Base")
            
            with gr.Accordion("📂 MODELOS (CLI Default)", open=True):
                # Usamos los argumentos CLI como valores por defecto
                u_mod = gr.Dropdown(get_files("unet"), label="UNET", value=args.unet, allow_custom_value=True)
                v_mod = gr.Dropdown(get_files("vae"), label="VAE Video", value=args.vae, allow_custom_value=True)
                va_mod = gr.Dropdown(get_files("vae"), label="VAE Audio", value=args.vae_audio, allow_custom_value=True)
                c1_mod = gr.Dropdown(get_files("clip"), label="CLIP 1", value=args.clip1, allow_custom_value=True)
                c2_mod = gr.Dropdown(get_files("clip"), label="CLIP 2", value=args.clip2, allow_custom_value=True)

            with gr.Accordion("🎨 LoRAs", open=False):
                loras = get_files("loras")
                l1 = gr.Dropdown(loras, label="L1", value=args.lora1, allow_custom_value=True)
                l1s = gr.Slider(-2, 2, args.lora1_st, label="Peso")
                
                l2 = gr.Dropdown(loras, label="L2", value=args.lora2, allow_custom_value=True)
                l2s = gr.Slider(-2, 2, args.lora2_st, label="Peso")
                
                l3 = gr.Dropdown(loras, label="L3", value=args.lora3, allow_custom_value=True)
                l3s = gr.Slider(-2, 2, args.lora3_st, label="Peso")
                
            btn_run = gr.Button("✨ GENERAR", variant="primary", size="lg")

        with gr.Column(scale=3):
            res = gr.Dropdown(["832x480 (16:9)", "1024x1024 (1:1)", "1280x720 (HD)", "480x832 (9:16)"], value="832x480 (16:9)", label="Res")
            steps = gr.Slider(1, 50, 20, step=1, label="Pasos")
            with gr.Row():
                sec = gr.Slider(1, 20, 5, step=1, label="Segundos")
                fps = gr.Slider(15, 60, 24, step=1, label="FPS")
                seed = gr.Number(label="Seed", value=2024, precision=0)
            st = gr.Slider(0, 1, 0.8, label="Fuerza Imagen")
            out_vid = gr.Video(); out_file = gr.File(); out_log = gr.Textbox(lines=5)

    def refresh():
        return (
            gr.update(choices=get_files("unet")), 
            gr.update(choices=get_files("vae")), 
            gr.update(choices=get_files("vae")), # Para audio VAE
            gr.update(choices=get_files("clip")), 
            gr.update(choices=get_files("clip")), 
            gr.update(choices=get_files("loras")), 
            gr.update(choices=get_files("loras")), 
            gr.update(choices=get_files("loras"))
        )

    btn_ref.click(refresh, None, [u_mod, v_mod, va_mod, c1_mod, c2_mod, l1, l2, l3])
    btn_mon.click(stats, None, monitor)
    btn_clr.click(lambda: (clear_all(), "🧹 Limpio"), None, out_log)
    
    btn_run.click(
        generate_process,
        [prompt, img, st, seed, steps, res, sec, fps, l1, l1s, l2, l2s, l3, l3s, u_mod, v_mod, va_mod, c1_mod, c2_mod],
        [out_log, out_vid, out_file]
    )

if __name__ == "__main__":
    demo.queue().launch(share=args.share, server_port=args.port, debug=True)
