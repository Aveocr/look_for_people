#!/usr/bin/env python3
"""
Search for a person by photos from a target folder against a gallery of persons.

Usage:
  python search_target.py --onnx model.onnx --target target_dir --gallery data/persons --k 5

This script loads an ONNX ReID model, computes average embeddings for each person
folder in the gallery, computes an average embedding for the target images, and
prints top-K matching person folders with similarity scores.
"""
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T


TRANSFORM = T.Compose([
    T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    )
])


def preprocess(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    t = TRANSFORM(img)
    return t.numpy()


def preprocess_batch(paths):
    return np.stack([preprocess(p) for p in paths]).astype(np.float32)


class ONNXModel:
    def __init__(self, onnx_path: str, use_cuda: bool = False):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)

    def infer(self, imgs_np: np.ndarray) -> np.ndarray:
        # imgs_np: [B, 3, 256, 128] float32
        out = self.session.run(["embedding"], {"image": imgs_np})[0]
        return out

    def close(self):
        try:
            del self.session
        except Exception:
            pass


class TRTModel:
    """Simple TensorRT engine wrapper using pycuda."""

    def __init__(self, trt_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self._trt = trt
        self._cuda = cuda

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.engine = engine

        # allocate buffers
        self._inp = None
        self._out = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            mode = engine.get_tensor_mode(name)
            max_shape = engine.get_tensor_profile_shape(name, 0)[2]
            host_buf = cuda.pagelocked_empty(int(np.prod(max_shape)), dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            entry = {"name": name, "host": host_buf, "device": dev_buf, "max_shape": max_shape}
            if mode == trt.TensorIOMode.INPUT:
                self._inp = entry
            else:
                self._out = entry

        self.emb_dim = int(self._out["max_shape"][-1])

    def infer(self, imgs_np: np.ndarray) -> np.ndarray:
        b = imgs_np.shape[0]
        # set dynamic shape
        self.context.set_input_shape("image", imgs_np.shape)

        np.copyto(self._inp["host"][:imgs_np.size], imgs_np.ravel())
        self._cuda.memcpy_htod_async(self._inp["device"], self._inp["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self._cuda.memcpy_dtoh_async(self._out["host"], self._out["device"], self.stream)
        self.stream.synchronize()
        return self._out["host"][:b * self.emb_dim].reshape(b, self.emb_dim).copy()

    def close(self):
        try:
            for buf in [self._inp, self._out]:
                if buf and "device" in buf:
                    buf["device"].free()
        except Exception:
            pass
        try:
            del self.context
            del self.engine
        except Exception:
            pass


def normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def build_gallery_embeddings(gallery_dir: str, model: ONNXModel, batch_size: int = 32):
    """Assumes gallery_dir contains subfolders per person (pid) with images."""
    pids = []
    emb_list = []
    rep_image = []

    subdirs = sorted([d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))])
    for d in subdirs:
        folder = os.path.join(gallery_dir, d)
        imgs = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not imgs:
            continue

        # compute embeddings in batches
        embs = []
        for i in range(0, len(imgs), batch_size):
            batch_paths = imgs[i:i+batch_size]
            batch = preprocess_batch(batch_paths)
            e = model.infer(batch)
            embs.append(e)
        embs = np.concatenate(embs, axis=0)
        embs = normalize(embs)
        mean_emb = embs.mean(axis=0, keepdims=True)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        pids.append(d)
        emb_list.append(mean_emb[0])
        rep_image.append(imgs[0])

    if not emb_list:
        return [], np.zeros((0, 512), dtype=np.float32), []

    emb_matrix = np.stack(emb_list, axis=0)
    return pids, emb_matrix, rep_image


def build_target_embedding(target_dir: str, model: ONNXModel, batch_size: int = 32):
    imgs = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not imgs:
        raise ValueError(f"No images found in target dir: {target_dir}")

    embs = []
    for i in range(0, len(imgs), batch_size):
        batch_paths = imgs[i:i+batch_size]
        batch = preprocess_batch(batch_paths)
        e = model.infer(batch)
        embs.append(e)
    embs = np.concatenate(embs, axis=0)
    embs = normalize(embs)
    mean_emb = embs.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    return mean_emb, imgs


def search(target_emb: np.ndarray, gallery_embs: np.ndarray, topk: int = 5):
    sims = gallery_embs @ target_emb
    idx = np.argsort(sims)[::-1][:topk]
    return idx, sims[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Path to ONNX model (with 'embedding' output)")
    parser.add_argument("--trt", help="Path to TensorRT engine (.trt). If provided, TRT is used")
    parser.add_argument("--target", required=True, help="Target images folder (all images averaged)")
    parser.add_argument("--gallery", required=True, help="Gallery root folder with person subfolders")
    parser.add_argument("--k", type=int, default=5, help="Top-K matches to show")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if args.trt:
        model = TRTModel(args.trt)
    elif args.onnx:
        model = ONNXModel(args.onnx, use_cuda=args.use_cuda)
    else:
        raise ValueError("Either --onnx or --trt must be provided")

    print("Building gallery embeddings...")
    pids, gallery_embs, rep_images = build_gallery_embeddings(args.gallery, model, batch_size=args.batch_size)
    if gallery_embs.shape[0] == 0:
        print("No gallery embeddings found. Exiting.")
        return

    print(f"Loaded {len(pids)} gallery persons")

    print("Building target embedding (averaging target images)...")
    target_emb, target_imgs = build_target_embedding(args.target, model, batch_size=args.batch_size)

    idx, scores = search(target_emb, gallery_embs, topk=args.k)

    print("Top matches:")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        pid = pids[i]
        rep = rep_images[i]
        print(f"{rank}. PID={pid}  score={float(s):.4f}  rep_image={rep}")

    model.close()


if __name__ == "__main__":
    main()
