import os
import json
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from search_target import (
    ONNXModel,
    TRTModel,
    build_gallery_embeddings,
    build_target_embedding,
    search,
)

BASE_DIR = Path(__file__).parent
app = FastAPI()

# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Defaults from environment
DEFAULT_ONNX = os.environ.get("MODEL_ONNX")
DEFAULT_TRT = os.environ.get("MODEL_TRT")
DEFAULT_GALLERY = os.environ.get("GALLERY_DIR", "data/persons")
DEFAULT_TARGET = os.environ.get("TARGET_DIR", "target")
USE_CUDA = os.environ.get("USE_CUDA", "false").lower() in ("1", "true", "yes")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Read and render template manually
    with open(BASE_DIR / "templates" / "index.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    # Simple template substitution
    html = html.replace("{{ default_onnx or '' }}", DEFAULT_ONNX or "")
    html = html.replace("{{ default_trt or '' }}", DEFAULT_TRT or "")
    html = html.replace("{{ default_gallery }}", DEFAULT_GALLERY)
    html = html.replace("{{ default_target }}", DEFAULT_TARGET)
    
    return HTMLResponse(content=html)


def _load_model(use_trt_path, trt_path, onnx_path, use_cuda_flag):
    if use_trt_path and trt_path:
        return TRTModel(trt_path)
    if onnx_path:
        return ONNXModel(onnx_path, use_cuda=use_cuda_flag)
    if DEFAULT_TRT:
        try:
            return TRTModel(DEFAULT_TRT)
        except Exception:
            pass
    if DEFAULT_ONNX:
        return ONNXModel(DEFAULT_ONNX, use_cuda=use_cuda_flag)
    raise ValueError("No model specified")


@app.post("/run", response_class=HTMLResponse)
async def run_action(request: Request):
    form = await request.form()
    action = form.get("action")
    gallery = form.get("gallery") or DEFAULT_GALLERY
    target = form.get("target") or DEFAULT_TARGET
    onnx_path = form.get("onnx_path")
    trt_path = form.get("trt_path")
    use_trt = bool(trt_path)
    use_cuda_flag = form.get("use_cuda") == "on"

    # initialize model
    try:
        model = _load_model(use_trt, trt_path, onnx_path, use_cuda_flag)
    except Exception as e:
        return HTMLResponse(content=f"Model load error: {e}", status_code=400)

    result = {}

    # Build gallery embeddings once
    pids, gallery_embs, rep_images = build_gallery_embeddings(gallery, model)
    result["gallery_count"] = len(pids)

    if action == "search_target":
        try:
            target_emb, target_imgs = build_target_embedding(target, model)
        except Exception as e:
            model.close()
            return HTMLResponse(content=f"Target build error: {e}", status_code=400)

        idx, scores = search(target_emb, gallery_embs, topk=10)
        matches = []
        for i, s in zip(idx, scores):
            matches.append({"pid": pids[int(i)], "score": float(s), "rep": rep_images[int(i)]})
        result["matches"] = matches

    elif action == "search_all":
        # naive all-vs-all: for each pid, search others with threshold
        threshold = float(form.get("threshold") or 0.8)
        matches = []
        for i, emb in enumerate(gallery_embs):
            sims = gallery_embs @ emb
            # ignore self
            sims[i] = -1.0
            idxs = list((sims >= threshold).nonzero()[0])
            for j in idxs:
                matches.append({"a": pids[i], "b": pids[int(j)], "score": float(sims[int(j)])})
        result["matches"] = matches

    elif action == "initialize_people":
        # Save gallery embeddings to JSON (init for tracking)
        out = []
        for pid, emb, rep in zip(pids, gallery_embs, rep_images):
            out.append({"pid": pid, "rep_image": rep, "emb_mean": emb.tolist()})
        os.makedirs("data", exist_ok=True)
        with open("data/people_init.json", "w", encoding="utf-8") as f:
            json.dump(out, f)
        result["saved"] = "data/people_init.json"

    else:
        model.close()
        return HTMLResponse(content="Unknown action", status_code=400)

    model.close()
    
    # Render results template manually
    with open(BASE_DIR / "templates" / "results.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    # Replace result dict with JSON
    html = html.replace("{{ result | tojson(indent=2, ensure_ascii=False) }}", json.dumps(result, indent=2, ensure_ascii=False))
    
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
