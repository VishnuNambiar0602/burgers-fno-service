"""Stage and upload the app to a Hugging Face Space (Docker SDK). Run by the USER, who owns the token.

  python scripts/deploy_hf.py --dry-run     # stages files and lists them; reads no token, uses no network
  python scripts/deploy_hf.py               # uploads (needs HF_TOKEN, HF_USERNAME in .env or the environment)

Reads only HF_TOKEN, HF_USERNAME, HF_SPACE_NAME from .env. The token is never printed.
"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STAGE = os.path.join(ROOT, "build", "space_stage")
FILES = {                                   # staged path -> source path (relative to repo root)
    "Dockerfile": "deploy/Dockerfile", "README.md": "deploy/SPACE_README.md",
    "requirements-serve.txt": "requirements-serve.txt",
    "app/__init__.py": "app/__init__.py", "app/main.py": "app/main.py",
    "src/__init__.py": "src/__init__.py", "src/models.py": "src/models.py",
    "src/inference.py": "src/inference.py", "src/solver.py": "src/solver.py",
    "model/fno_s1.pt": "model/fno_s1.pt", "model/ood_stats.json": "model/ood_stats.json",
    "model/model_card.json": "model/model_card.json",
}
EXTRA_DIRS = ["app/static"]                 # added in M5 when it exists


def read_env():
    env = {}
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return {k: os.environ.get(k) or env.get(k, "") for k in ("HF_TOKEN", "HF_USERNAME", "HF_SPACE_NAME")}


def stage():
    shutil.rmtree(STAGE, ignore_errors=True)
    total = 0
    for dst, src in FILES.items():
        s, d = os.path.join(ROOT, src), os.path.join(STAGE, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        total += os.path.getsize(d)
        print(f"  {dst:28s} {os.path.getsize(d):>10,d} bytes")
    for dd in EXTRA_DIRS:
        if os.path.isdir(os.path.join(ROOT, dd)):
            shutil.copytree(os.path.join(ROOT, dd), os.path.join(STAGE, dd))
            print(f"  {dd}/ (directory)")
    print(f"staged {len(FILES)} files, {total / 1e6:.2f} MB -> {os.path.relpath(STAGE, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the Space as private (default: public)")
    a = ap.parse_args()
    print("Staging:")
    stage()
    if a.dry_run:
        print("dry run: nothing uploaded, token not read.")
        return
    cfg = read_env()
    missing = [k for k in ("HF_TOKEN", "HF_USERNAME", "HF_SPACE_NAME") if not cfg[k]]
    if missing:
        sys.exit(f"Missing in .env or environment: {', '.join(missing)}")
    token, repo = cfg["HF_TOKEN"], f"{cfg['HF_USERNAME']}/{cfg['HF_SPACE_NAME']}"
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
        print(f"authenticated as: {who}")
        if who.lower() != cfg["HF_USERNAME"].lower():
            print(f"NOTE: token belongs to '{who}' but HF_USERNAME is '{cfg['HF_USERNAME']}'.")
        api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker", exist_ok=True, private=a.private)
        print(f"Space ready: https://huggingface.co/spaces/{repo} (private={a.private})")
        api.upload_folder(folder_path=STAGE, repo_id=repo, repo_type="space", commit_message="Deploy burgers-fno-service skeleton")
        print("uploaded. Waiting for the build (up to 20 min)...")
        last, t0 = None, time.time()
        while time.time() - t0 < 1200:
            stage_now = str(api.get_space_runtime(repo).stage)
            if stage_now != last:
                print(f"  {time.strftime('%H:%M:%S')} stage: {stage_now}", flush=True)
                last = stage_now
            if any(s in stage_now.upper() for s in ("RUNNING", "ERROR", "STOPPED", "PAUSED")):
                break
            time.sleep(10)
        print(f"URL (when RUNNING): https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space")
    except Exception as e:                      # never leak the token in an error message
        print("FAILED:", type(e).__name__, str(e).replace(token, "***")[:600])
        sys.exit(1)


if __name__ == "__main__":
    main()
