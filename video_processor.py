import os
import cv2
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

from search_target import ONNXModel, TRTModel, build_gallery_embeddings, normalize


class SimplePersonGallery:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.persons = {}  # pid -> {'emb': emb, 'count': n}
        self._next_id = 1

    def match_or_add(self, emb: np.ndarray):
        if not self.persons:
            pid = str(self._next_id)
            self.persons[pid] = {'emb': emb.copy(), 'count': 1}
            self._next_id += 1
            return pid

        ids = list(self.persons.keys())
        matrix = np.stack([self.persons[i]['emb'] for i in ids], axis=0)
        sims = matrix @ emb
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            pid = ids[best]
            # update mean
            p = self.persons[pid]
            n = p['count']
            updated = (p['emb'] * n + emb) / (n + 1)
            p['emb'] = updated / (np.linalg.norm(updated) + 1e-8)
            p['count'] = n + 1
            return pid
        else:
            pid = str(self._next_id)
            self.persons[pid] = {'emb': emb.copy(), 'count': 1}
            self._next_id += 1
            return pid


def extract_face_crops(src_dir: str,
                       dst_dir: str,
                       face_model: str | None = None,
                       conf: float = 0.35,
                       iou: float = 0.4,
                       min_size: int = 32):
    """Extract faces from a folder of images using YOLO face detector."""
    from ultralytics import YOLO

    detector = YOLO(face_model or "yolov8n-face.pt")
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(src.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = detector(img, conf=conf, iou=iou)
        crop_index = 0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue
                crop = img[y1:y2, x1:x2]
                out_path = dst / f"{img_path.stem}_face_{crop_index}.jpg"
                cv2.imwrite(str(out_path), crop)
                crop_index += 1

    return str(dst)


def process_video(video_path: str,
                  model_backend: object,
                  out_dir: str = "data",
                  skip_frames: int = 1,
                  min_crop_px: int = 32,
                  threshold: float = 0.65,
                  method: str = "body",
                  face_model: str | None = None,
                  body_model: str | None = None):
    """
    Simple video processing: detect crops with YOLO (body or face), compute embeddings,
    assign person ids online and save crops + JSON.

    Returns path to results JSON.
    """
    out_dir = Path(out_dir)
    persons_dir = out_dir / "persons"
    json_dir = out_dir / "json"
    persons_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a YOLO detector (ultralytics)
    detector = None
    detector_type = "none"
    try:
        from ultralytics import YOLO
        if method == "face":
            model_path = face_model or "yolov8n-face.pt"
            detector = YOLO(model_path)
            detector_type = "face"
        else:
            model_path = body_model or "yolov8m.pt"
            detector = YOLO(model_path)
            detector_type = "body"
    except Exception as exc:
        if method == "face":
            raise RuntimeError(f"Face YOLO detector load failed: {exc}") from exc
        detector = None
        detector_type = "none"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    gallery = SimplePersonGallery(threshold=threshold)

    frame_idx = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if skip_frames > 1 and (frame_idx % skip_frames) != 0:
            continue

        h, w = frame.shape[:2]
        crops = []
        if detector is not None:
            conf = 0.35 if detector_type == "face" else 0.45
            if detector_type == "face":
                results = detector(frame, conf=conf, iou=0.4)
            else:
                results = detector(frame, conf=conf, iou=0.4, classes=[0])
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    crops.append((crop, (x1, y1, x2, y2)))
        else:
            # fallback: take center crop
            cx1 = max(0, w // 4)
            cy1 = max(0, h // 4)
            cx2 = min(w, cx1 + w // 2)
            cy2 = min(h, cy1 + h // 2)
            crop = frame[cy1:cy2, cx1:cx2]
            crops.append((crop, (cx1, cy1, cx2, cy2)))

        if not crops:
            continue

        # Preprocess crops and run model
        try:
            import torchvision.transforms as T
            from PIL import Image
            TRANSFORM = T.Compose([
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275,  0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            imgs = []
            for crop, bbox in crops:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                imgs.append(TRANSFORM(pil).numpy())
            imgs_np = np.stack(imgs).astype(np.float32)
            embs = model_backend.infer(imgs_np)
            embs = normalize(embs)
        except Exception:
            # skip if model fails
            embs = np.zeros((len(crops), 512), dtype=np.float32)

        for (crop, bbox), emb in zip(crops, embs):
            pid = gallery.match_or_add(emb)
            person_folder = persons_dir / pid
            person_folder.mkdir(parents=True, exist_ok=True)
            fname = f"{uuid.uuid4().hex}.png"
            fpath = person_folder / fname
            cv2.imwrite(str(fpath), crop)
            saved.append({"pid": pid, "file": str(fpath), "bbox": bbox, "frame": frame_idx})

    cap.release()

    out_path = json_dir / f"persons_{int(time.time())}.json"
    import json as _json
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump({
            "method": method,
            "detector": detector_type,
            "video": str(video_path),
            "saved": saved,
        }, f, ensure_ascii=False, indent=2)

    return str(out_path)
