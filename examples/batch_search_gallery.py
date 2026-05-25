import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from search_target import ONNXModel, TRTModel, build_gallery_embeddings, build_target_embedding, search


def detect_face_presence(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
    return len(faces) > 0


def make_result_grid(target_img_path, match_paths, out_path):
    # 3x3 grid, center is target
    imgs = []
    # load target
    t = cv2.imread(str(target_img_path))
    if t is None:
        t = np.zeros((256,256,3), dtype=np.uint8)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    imgs.append(t)

    # ensure we have 8 matches
    for p in match_paths[:8]:
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros_like(t)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)

    while len(imgs) < 9:
        imgs.append(np.zeros_like(t))

    # layout: we want center index 4 to be target; rearrange accordingly
    # we'll put matches around center: positions 0-3,5-8 are matches
    center = imgs[0]
    matches = imgs[1:9]
    grid = [matches[0], matches[1], matches[2],
            matches[3], center, matches[4],
            matches[5], matches[6], matches[7]]

    fig, axes = plt.subplots(3,3, figsize=(8,8))
    for ax, im in zip(axes.flatten(), grid):
        ax.imshow(im)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True, help='ONNX model path')
    p.add_argument('--gallery', default='data/outputs/img', help='Gallery dir (per-person ids)')
    p.add_argument('--target', required=True, help='Target dir with subfolders (one per person)')
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--out', default='results', help='Output folder')
    p.add_argument('--use_cuda', action='store_true')
    args = p.parse_args()

    onnx_path = args.onnx
    gallery_dir = Path(args.gallery)
    target_dir = Path(args.target)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ONNXModel(onnx_path, use_cuda=args.use_cuda)

    print('Building gallery embeddings...')
    pids, gallery_embs, rep_images = build_gallery_embeddings(str(gallery_dir), model, batch_size=32)

    rows = []
    for person_folder in sorted(target_dir.iterdir()):
        if not person_folder.is_dir():
            continue
        print('Processing target person:', person_folder.name)
        # Build target embedding from this person's folder
        target_emb = build_target_embedding(str(person_folder), model, batch_size=16)

        results = search(gallery_embs, target_emb, topk=args.topk)

        # pick representative image for target (first file)
        target_img = next(person_folder.glob('*.*'), None)
        if target_img is None:
            continue

        match_paths = []
        for rank, (idx, score) in enumerate(zip(*results), start=1):
            idx = int(idx)
            pid = pids[idx]
            img_path = rep_images[idx]
            ts = datetime.now().isoformat()
            face_in_target = detect_face_presence(target_img)
            face_in_match = detect_face_presence(img_path)
            rows.append({
                'query_person': person_folder.name,
                'match_pid': pid,
                'match_img': str(img_path),
                'score': float(score),
                'time': ts,
                'video': Path(img_path).stem,
                'face_in_target': face_in_target,
                'face_in_match': face_in_match,
            })
            match_paths.append(img_path)

        # build grid image
        grid_path = out_dir / f"{person_folder.name}_grid.jpg"
        make_result_grid(target_img, match_paths, str(grid_path))

    df = pd.DataFrame(rows)
    excel_path = out_dir / 'matches.xlsx'
    df.to_excel(excel_path, index=False)
    print('Saved results to', excel_path)

    # cleanup
    try:
        getattr(model, 'close', lambda: None)()
    except Exception:
        pass


if __name__ == '__main__':
    main()
