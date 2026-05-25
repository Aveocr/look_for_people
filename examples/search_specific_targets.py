import argparse
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from search_target import ONNXModel, build_gallery_embeddings, build_target_embedding, search


def make_grid_and_save(target_img_path, match_paths, out_path):
    # reuse similar logic to batch script
    imgs = []
    t = cv2.imread(str(target_img_path))
    if t is None:
        t = np.zeros((256,256,3), dtype=np.uint8)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    imgs.append(t)
    for p in match_paths[:8]:
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros_like(t)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
    while len(imgs) < 9:
        imgs.append(np.zeros_like(t))
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
    p.add_argument('--onnx', required=True)
    p.add_argument('--gallery', default='data/outputs/img')
    p.add_argument('--targets', nargs='*', help='List of target image files to search')
    p.add_argument('--out', default='results_specific')
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--use_cuda', action='store_true')
    args = p.parse_args()

    gallery_dir = Path(args.gallery)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ONNXModel(args.onnx, use_cuda=args.use_cuda)
    pids, gallery_embs, rep_images = build_gallery_embeddings(str(gallery_dir), model, batch_size=32)

    rows = []
    for tpath in args.targets:
        tfile = Path(tpath)
        if not tfile.exists():
            print('Target not found:', tpath)
            continue
        # create temp folder with this file
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            shutil.copy(tfile, tmp / tfile.name)
            target_emb = build_target_embedding(str(tmp), model, batch_size=1)
            results = search(gallery_embs, target_emb, topk=args.topk)
            match_paths = []
            for idx, score in results:
                match_paths.append(rep_images[idx])
                rows.append({
                    'query_file': str(tfile),
                    'match_img': str(rep_images[idx]),
                    'score': float(score),
                    'time': datetime.now().isoformat(),
                })
            grid_path = out_dir / f"{tfile.stem}_grid.jpg"
            make_grid_and_save(tfile, match_paths, str(grid_path))

    df = pd.DataFrame(rows)
    excel_path = out_dir / 'matches_specific.xlsx'
    df.to_excel(excel_path, index=False)
    print('Saved specific results to', excel_path)
    try:
        getattr(model, 'close', lambda: None)()
    except Exception:
        pass


if __name__ == '__main__':
    main()
