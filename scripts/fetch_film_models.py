"""Place the AI Film models on the network volume (mounted at /workspace on a pod).

Wan2.2-I2V-A14B transformers are converted fp32 -> bf16 one shard at a time (the fp32 release is
126 GB; bf16 is what the pipeline runs). Qwen-Image-Edit-2511 is copied as published. Revisions pinned.
Run: ROOT=/workspace python3 scripts/fetch_film_models.py  (needs ~135 GB free on the volume).
"""
import os, json, time, shutil, glob, subprocess
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
ROOT = os.environ.get("ROOT", "/workspace"); LOG = os.environ.get("LOG", "/tmp/fetch_film_models.log")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(LOG, "a").write(s + "\n")
def df():
    t, u, f = shutil.disk_usage(ROOT); return f"vol used={u/1e9:.1f}GB free={f/1e9:.1f}GB total={t/1e9:.1f}GB"
def du(p): return round(sum(os.path.getsize(x) for x in glob.glob(p + "/**/*", recursive=True) if os.path.isfile(x)) / 1e9, 2)
QWEN = ("Qwen/Qwen-Image-Edit-2511", "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9", "qwen-image-edit-2511")
WAN = ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "596658fd9ca6b7b71d5057529bbf319ecbc61d74", "wan2.2-i2v-a14b")
LORAS = [
    ("lightx2v/Wan2.2-Lightning", "18bccf8884ec0a078eed79785eb4ef13ea16ce1e", "wan2.2-i2v-a14b",
     ["Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
      "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"]),
    ("lightx2v/Qwen-Image-Edit-2511-Lightning", "107d238d68b8387e32843d0b4a2e24ceb0d3ad86", "qwen-image-edit-2511",
     ["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"]),
]
try:
    log("START", df())
    repo, rev, d = WAN; dest = f"{ROOT}/{d}"
    if os.path.exists(f"{dest}/AIMX_SOURCE.json"):
        log("wan already placed")
    else:
      snapshot_download(repo, revision=rev, local_dir=dest, ignore_patterns=["transformer/*.safetensors", "transformer_2/*.safetensors", "assets/*", "examples/*", "*.mp4", "*.gif", "*.png", "*.jpg"])
      log("wan base done", du(dest), df())
      import torch
      from safetensors.torch import load_file, save_file
      for f in sorted(x for x in list_repo_files(repo, revision=rev) if x.split("/")[0] in ("transformer", "transformer_2") and x.endswith(".safetensors")):
          out = f"{dest}/{f}"
          if os.path.exists(out): log("skip", f); continue
          tmp = hf_hub_download(repo, f, revision=rev, local_dir="/tmp/dl")
          sd = load_file(tmp)
          sd = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in sd.items()}
          os.makedirs(os.path.dirname(out), exist_ok=True)
          save_file(sd, out + ".incomplete", metadata={"format": "pt"}); os.replace(out + ".incomplete", out)
          del sd; os.remove(tmp)
          log("bf16", f, round(os.path.getsize(out) / 1e9, 2), "GB", df())
      for sub in ("transformer", "transformer_2"):
          idx = f"{dest}/{sub}/diffusion_pytorch_model.safetensors.index.json"
          if os.path.exists(idx):
              j = json.load(open(idx)); j.setdefault("metadata", {})["total_size"] = sum(os.path.getsize(f"{dest}/{sub}/{x}") for x in set(j["weight_map"].values()))
              json.dump(j, open(idx, "w"))
      shutil.rmtree(f"{dest}/.cache", ignore_errors=True)
      open(f"{dest}/AIMX_SOURCE.json", "w").write(json.dumps({"repo": repo, "revision": rev, "transformers_dtype": "bfloat16 (converted from fp32)"}))
    log("WAN DONE", du(dest), df())
    repo, rev, d = QWEN; dest = f"{ROOT}/{d}"
    if not os.path.exists(f"{dest}/AIMX_SOURCE.json"):
        snapshot_download(repo, revision=rev, local_dir=dest, ignore_patterns=["*.mp4", "*.gif", "*.png", "*.jpg", "assets/*"])
        shutil.rmtree(f"{dest}/.cache", ignore_errors=True)
        open(f"{dest}/AIMX_SOURCE.json", "w").write(json.dumps({"repo": repo, "revision": rev}))
    log("QWEN DONE", du(dest), df())
    for repo, rev, d, files in LORAS:
        for src in files:
            out = f"{ROOT}/{d}/lightning/{os.path.basename(src)}"
            if os.path.exists(out): continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = hf_hub_download(repo, src, revision=rev, local_dir="/tmp/lora")
            shutil.copyfile(tmp, out + ".incomplete"); os.replace(out + ".incomplete", out); os.remove(tmp)
            log("lora", out, round(os.path.getsize(out) / 1e9, 2), "GB")
    log("ALL DONE", df())
except Exception as e:
    import traceback; log("ERROR", repr(e), traceback.format_exc()[-1500:])
