"""

使い方例:
python examples/otani.py --image0 assets/設計図/table_parts.png --image1 assets/設計図/table.png --manual sam3/manual.txt --prompts0 "wooden board" --prompts1 "wooden board" "table"
"""

import argparse
# import itertools
import json
import math
import os
import sys
from pathlib import Path
import subprocess
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import cv2
ROOT_DIR = Path(__file__).resolve().parent
SAM3_LOCAL_ROOT = ROOT_DIR / "sam3"
if SAM3_LOCAL_ROOT.exists():
    sys.path.insert(0, str(SAM3_LOCAL_ROOT))

import sam3
import torch
import timm
from PIL import Image, ImageOps
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision import transforms as T


def load_models():
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)

    dino_device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
    dino_model = dino_model.to(dino_device).eval()
    dino_preprocess = T.Compose(
        [
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, dino_model, dino_preprocess, dino_device


def run_prompts(
    image_path,
    prompts,
    run_dir,
    model,
    dino_model,
    dino_preprocess,
    dino_device,
    confidence=0.5,
    annot_name: str | None = None,
    draw_index: bool = True,
    use_id_mask: bool = False,
):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # EXIF情報に基づいて画像を正しい向きに回転
    width, height = image.size

    # If ID masks already exist, skip SAM3 and use precomputed masks/boxes.
    id_mask_dir = run_dir / f"{Path(image_path).stem}" / "IdMask"
    if use_id_mask and id_mask_dir.exists():
        base_dir = run_dir / f"{Path(image_path).stem}"
        if base_dir.exists():
            import shutil
            for child in base_dir.iterdir():
                if child.name == "IdMask":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        bbox_root = base_dir / "BoundingBox"
        bbox_root.mkdir(parents=True, exist_ok=True)
        box_map = None
        box_path = id_mask_dir / "mask_boxes.json"
        if box_path.exists():
            try:
                payload = json.loads(box_path.read_text())
                box_map = payload.get("boxes")
            except Exception:
                box_map = None
        id_mask_png = id_mask_dir / "id_mask.png"
        id_mask_npy = id_mask_dir / "id_mask.npy"
        id_mask_map = id_mask_dir / "id_mask_map.json"
        total_boxes = 0
        rgb_np = np.array(image, dtype=np.uint8)
        # Prefer full-size ID mask decoding when available.
        if (id_mask_npy.exists() or id_mask_png.exists()) and id_mask_map.exists():
            def _encode_id_color(part_id: int):
                h = (part_id * 2654435761) & 0xFFFFFFFF
                r = 64 + (h & 0xFF) % 192
                g = 64 + ((h >> 8) & 0xFF) % 192
                b = 64 + ((h >> 16) & 0xFF) % 192
                return r, g, b

            id_map = json.loads(id_mask_map.read_text())
            part_ids = sorted(id_map.values())
            id_arr = None
            if id_mask_npy.exists():
                try:
                    id_arr = np.load(id_mask_npy)
                except Exception:
                    id_arr = None
            id_img = None
            if id_arr is None and id_mask_png.exists():
                id_img = np.array(Image.open(id_mask_png).convert("RGB"), dtype=np.int16)
            tol = 2
            # If colors were altered by color management, map closest colors.
            part_color_map = {}
            if id_img is not None:
                uniq = np.unique(id_img.reshape(-1, 3), axis=0)
                uniq = [tuple(map(int, c)) for c in uniq if not (c[0] == 0 and c[1] == 0 and c[2] == 0)]
                expected = {pid: _encode_id_color(pid) for pid in part_ids}
                if uniq:
                    uniq_arr = np.array(uniq, dtype=np.int32)
                    if len(uniq) >= len(part_ids):
                        remaining = set(range(len(uniq)))
                        for pid in part_ids:
                            target = np.array(expected[pid], dtype=np.int32)
                            if not remaining:
                                break
                            idxs = np.array(sorted(remaining), dtype=np.int32)
                            cand = uniq_arr[idxs]
                            dists = np.sum((cand - target) ** 2, axis=1)
                            best_local = int(idxs[int(np.argmin(dists))])
                            part_color_map[pid] = uniq[best_local]
                            remaining.remove(best_local)
                    else:
                        # Fewer unique colors than part IDs: map colors to closest IDs and use subset.
                        remaining_pids = set(part_ids)
                        for color in uniq:
                            target = np.array(color, dtype=np.int32)
                            pid_list = sorted(remaining_pids)
                            exp_arr = np.array([expected[pid] for pid in pid_list], dtype=np.int32)
                            dists = np.sum((exp_arr - target) ** 2, axis=1)
                            best_idx = int(np.argmin(dists))
                            best_pid = pid_list[best_idx]
                            part_color_map[best_pid] = color
                            remaining_pids.remove(best_pid)
                        if remaining_pids:
                            print(f"[warn] id_mask.png has fewer colors than parts ({len(uniq)}/{len(part_ids)}); using subset")

            for prompt in prompts:
                prompt_dir = bbox_root / prompt.replace(" ", "_")
                prompt_dir.mkdir(parents=True, exist_ok=True)
                mask_dir = prompt_dir / "Masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                crop_dir = prompt_dir / "Masked_Crops"
                crop_dir.mkdir(parents=True, exist_ok=True)

                masks = []
                boxes_padded = []
                centers = []
                for local_idx, part_id in enumerate(part_ids):
                    if id_arr is not None:
                        mask = (id_arr == (part_id - 1))
                    else:
                        if part_color_map and part_id not in part_color_map:
                            continue
                        r, g, b = part_color_map.get(part_id, _encode_id_color(part_id))
                        mask = (
                            (np.abs(id_img[..., 0] - r) <= tol)
                            & (np.abs(id_img[..., 1] - g) <= tol)
                            & (np.abs(id_img[..., 2] - b) <= tol)
                        )
                    ys, xs = np.where(mask)
                    if ys.size == 0 or xs.size == 0:
                        continue
                    y0c, y1c = ys.min(), ys.max() + 1
                    x0c, x1c = xs.min(), xs.max() + 1
                    pad = 10
                    x0 = int(max(0, x0c - pad))
                    y0 = int(max(0, y0c - pad))
                    x1 = int(min(width, x1c + pad))
                    y1 = int(min(height, y1c + pad))
                    boxes_padded.append([x0, y0, x1, y1])
                    centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
                    m_full = (mask.astype(np.uint8) * 255)
                    masks.append(m_full)
                    Image.fromarray(m_full).save(mask_dir / f"mask_{local_idx}.png")

                    crop_rgb = rgb_np[y0c:y1c, x0c:x1c]
                    alpha = m_full[y0c:y1c, x0c:x1c]
                    crop_np = np.dstack([crop_rgb, alpha])
                    Image.fromarray(crop_np, mode="RGBA").save(crop_dir / f"mask_crop_{local_idx}.png")

                bbox_path = prompt_dir / "boxes.json"
                with open(bbox_path, "w") as f:
                    json.dump(boxes_padded, f)
                centers_path = prompt_dir / "centers.json"
                with open(centers_path, "w") as f:
                    json.dump(centers, f)
                # centers_txt = prompt_dir / "centers.txt"
                # with open(centers_txt, "w") as f:
                #     for i, (cx, cy) in enumerate(centers):
                #         f.write(f"パーツ {i}: x={cx:.6f}, y={cy:.6f}\n")
                feature_dir = prompt_dir / "BBox_Feature"
                feature_dir.mkdir(parents=True, exist_ok=True)
                for idx, box in enumerate(boxes_padded):
                    x0, y0, x1, y1 = box
                    mask_crop_path = crop_dir / f"mask_crop_{idx}.png"
                    if mask_crop_path.exists():
                        mc_rgba = Image.open(mask_crop_path).convert("RGBA")
                        bg = Image.new("RGBA", mc_rgba.size, (255, 255, 255, 255))
                        mc_rgb = Image.alpha_composite(bg, mc_rgba).convert("RGB")
                        crop = mc_rgb
                    else:
                        crop = image.crop((x0, y0, x1, y1)).convert("RGB")
                    feat_path = feature_dir / f"box_{idx}.pt"
                    with torch.no_grad():
                        dino_in = dino_preprocess(crop).unsqueeze(0).to(dino_device)
                        out = dino_model.forward_features(dino_in)
                        if isinstance(out, dict):
                            tokens = (
                                out.get("x_norm_patchtokens")
                                or out.get("x_norm_patch_tokens")
                                or out.get("patch_tokens")
                                or out.get("tokens")
                            )
                            if tokens is None:
                                tokens = out.get("x_norm_clstoken") or out.get("x_norm_cls_token") or out.get("cls_token")
                            if tokens is None:
                                raise ValueError("DINO features not found")
                        else:
                            tokens = out
                        if tokens.dim() == 3:  # [B, N, D]形式をチェック。もしそうならプーリングをし、１本の特徴ベクトルにして正規化する
                            pooled = tokens.mean(dim=1)
                        else:
                            pooled = tokens
                        dino_feat = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                        torch.save(dino_feat.cpu(), feat_path)
                total_boxes += len(boxes_padded)

            if total_boxes > 0:
                return base_dir, total_boxes
            print("[warn] id_mask.png decoding produced no boxes; fallback to mask_*.png")
        
        for prompt in prompts:
            prompt_dir = bbox_root / prompt.replace(" ", "_")
            prompt_dir.mkdir(parents=True, exist_ok=True)
            mask_dir = prompt_dir / "Masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            crop_dir = prompt_dir / "Masked_Crops"
            crop_dir.mkdir(parents=True, exist_ok=True)

            masks = []
            boxes_padded = []
            centers = []
            def _mask_index(p: Path):  # マスクファイル名の末尾の数字を取得
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return 10**9
            for mpath in sorted(id_mask_dir.glob("mask_*.png"), key=_mask_index):  # マスクファイルをソート
                m = np.array(Image.open(mpath).convert("L"))
                ys, xs = np.where(m > 0)
                if ys.size == 0 or xs.size == 0:  # マスクが空の場合はスキップ
                    continue
                mask_idx = int(mpath.stem.split("_")[-1])
                local_idx = len(boxes_padded)

                if box_map is not None and mask_idx < len(box_map) and box_map[mask_idx]:  # ボックスマップがある場合はボックスマップを使用
                    x0c, y0c, x1c, y1c = box_map[mask_idx]
                else:  # ボックスマップがない場合はマスクの範囲を使用
                    y0c, y1c = ys.min(), ys.max() + 1
                    x0c, x1c = xs.min(), xs.max() + 1

                raw_w = max(1, int(x1c - x0c))  # マスクの幅
                raw_h = max(1, int(y1c - y0c))  # マスクの高さ
                mh, mw = m.shape[:2]
                pad_x = 10  # パディング
                pad_y = 10  # パディング
                x0 = int(max(0, x0c - pad_x))   # パディングを考慮した左端
                y0 = int(max(0, y0c - pad_y))   # パディングを考慮した上端
                x1 = int(min(width, x0 + mw))   # パディングを考慮した右端
                y1 = int(min(height, y0 + mh))   # パディングを考慮した下端
                crop_rgb = rgb_np[y0c:y1c, x0c:x1c]
                alpha = (m > 0).astype(np.uint8) * 255  # マスクをアルファ（透過）に変換
                alpha = alpha[pad_y : pad_y + raw_h, pad_x : pad_x + raw_w]
                if alpha.shape[0] != crop_rgb.shape[0] or alpha.shape[1] != crop_rgb.shape[1]:  
                    alpha = alpha[: crop_rgb.shape[0], : crop_rgb.shape[1]]
                crop_np = np.dstack([crop_rgb, alpha])  # クロップをRGBAに変換（Aはマスクのアルファ）
                boxes_padded.append([x0, y0, x1, y1])  
                centers.append([(x0 + x1) / 2, (y0 + y1) / 2])  
                if m.shape[0] == height and m.shape[1] == width:  # マスクがフルサイズの場合はそのまま使用。ここいらなそう
                    m_full = m
                else:  # マスクがフルサイズでない場合は0でパディング
                    m_full = np.zeros((height, width), dtype=np.uint8)
                    x1p = min(width, x0 + mw)  # パディングを考慮した右端
                    y1p = min(height, y0 + mh)  # パディングを考慮した下端
                    m_full[y0:y1p, x0:x1p] = m[: y1p - y0, : x1p - x0]  # マスクをパディングした範囲にコピー
                masks.append(m_full)
                Image.fromarray(m_full).save(mask_dir / f"mask_{local_idx}.png")
                crop_name = f"mask_crop_{local_idx}.png"
                Image.fromarray(crop_np, mode="RGBA").save(crop_dir / crop_name)
                continue

            bbox_path = prompt_dir / "boxes.json"
            with open(bbox_path, "w") as f:
                json.dump(boxes_padded, f)
            centers_path = prompt_dir / "centers.json"
            with open(centers_path, "w") as f:
                json.dump(centers, f)
            centers_txt = prompt_dir / "centers.txt"
            with open(centers_txt, "w") as f:
                for i, (cx, cy) in enumerate(centers):
                    f.write(f"パーツ {i}: x={cx:.6f}, y={cy:.6f}\n")
            feature_dir = prompt_dir / "BBox_Feature"
            feature_dir.mkdir(parents=True, exist_ok=True)

            for idx, box in enumerate(boxes_padded):
                x0, y0, x1, y1 = box
                mask_crop_path = crop_dir / f"mask_crop_{idx}.png"
                if mask_crop_path.exists():
                    mc_rgba = Image.open(mask_crop_path).convert("RGBA")
                    bg = Image.new("RGBA", mc_rgba.size, (255, 255, 255, 255))
                    mc_rgb = Image.alpha_composite(bg, mc_rgba).convert("RGB")
                    crop = mc_rgb
                else:
                    crop = image.crop((x0, y0, x1, y1)).convert("RGB")

                feat_path = feature_dir / f"box_{idx}.pt"
                with torch.no_grad():
                    dino_in = dino_preprocess(crop).unsqueeze(0).to(dino_device)
                    out = dino_model.forward_features(dino_in)
                    if isinstance(out, dict):
                        tokens = (
                            out.get("x_norm_patchtokens")
                            or out.get("x_norm_patch_tokens")
                            or out.get("patch_tokens")
                            or out.get("tokens")
                        )
                        if tokens is None:
                            tokens = out.get("x_norm_clstoken") or out.get("x_norm_cls_token") or out.get("cls_token")
                        if tokens is None:
                            raise ValueError("DINO features not found")
                    else:
                        tokens = out
                    if tokens.dim() == 3:
                        pooled = tokens.mean(dim=1)
                    else:
                        pooled = tokens
                    dino_feat = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                    torch.save(dino_feat.cpu(), feat_path)

            total_boxes += len(boxes_padded)

        return base_dir, total_boxes
    processor = Sam3Processor(model, confidence_threshold=confidence)
    state = processor.set_image(image)

    base_dir = run_dir / f"{Path(image_path).stem}"
    if base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    bbox_root = base_dir / "BoundingBox"
    bbox_root.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        prompt_dir = bbox_root / prompt.replace(" ", "_")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = prompt_dir / "Masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = prompt_dir / "Masked_Crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        # 10px 拡張ボックスで保存
        boxes_xyxy = state["boxes"].cpu().tolist()
        pad = 10
        boxes_padded = []
        centers = []
        for x0, y0, x1, y1 in boxes_xyxy:
            x0 = int(max(0, x0 - pad))
            y0 = int(max(0, y0 - pad))
            x1 = int(min(width, x1 + pad))
            y1 = int(min(height, y1 + pad))
            boxes_padded.append([x0, y0, x1, y1])
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            centers.append([cx, cy])

        bbox_path = prompt_dir / "boxes.json"
        with open(bbox_path, "w") as f:
            json.dump(boxes_padded, f)
        centers_path = prompt_dir / "centers.json"
        with open(centers_path, "w") as f:
            json.dump(centers, f)
        centers_txt = prompt_dir / "centers.txt"
        with open(centers_txt, "w") as f:
            for i, (cx, cy) in enumerate(centers):
                f.write(f"パーツ {i}: x={cx:.6f}, y={cy:.6f}\n")
        # マスク保存（フルサイズ）
        masks = state.get("masks", None)
        if masks is not None:
            masks_np = masks.cpu().numpy()  # [N, H, W] or [N, 1, H, W]
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0, :, :]
            for idx, m in enumerate(masks_np):
                if m.ndim == 3:
                    # 予期せぬ形状の場合はチャンネル平均
                    m = m.mean(axis=0)
                m_bin = (m > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(m_bin, mode="L")
                mask_img.save(mask_dir / f"mask_{idx}.png")
                # マスクを適用したクロップ（輪郭でトリミング）
                ys, xs = np.where(m_bin > 0)
                if ys.size == 0 or xs.size == 0:
                    continue
                y0c, y1c = ys.min(), ys.max() + 1
                x0c, x1c = xs.min(), xs.max() + 1
                rgb_np = np.array(image, dtype=np.uint8)
                # RGBA にしてマスクをアルファに使用（背景透過）
                rgba = np.dstack([rgb_np, np.zeros_like(m_bin, dtype=np.uint8)])
                rgba[ys, xs, 3] = 255  # マスク部分のみ不透明
                crop_np = rgba[y0c:y1c, x0c:x1c]
                crop_img = Image.fromarray(crop_np, mode="RGBA")
                crop_img.save(crop_dir / f"mask_crop_{idx}.png")
        total_boxes += len(boxes_padded)

        # 検出結果の可視化を base_dir に保存
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        for idx, b in enumerate(boxes_padded):
            rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=0.5, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            ax.text(
                cx,
                cy,
                f"{idx}",
                color="red",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.1"),
            )
        ax.axis("off")
        plt.tight_layout()
        det_path = base_dir / f"det_{prompt_dir.name}.png"
        plt.savefig(det_path, dpi=200)
        plt.close(fig)
        print(f"検出結果画像保存: {det_path}")

        feature_dir = prompt_dir / "BBox_Feature"
        feature_dir.mkdir(parents=True, exist_ok=True)

        for idx, box in enumerate(boxes_padded):
            x0, y0, x1, y1 = box
            # 優先: マスクで輪郭トリミングしたクロップを DINO 入力に使用
            mask_crop_path = crop_dir / f"mask_crop_{idx}.png"
            if mask_crop_path.exists():
                mc_rgba = Image.open(mask_crop_path).convert("RGBA")
                # 白背景でアルファ合成して RGB 化
                bg = Image.new("RGBA", mc_rgba.size, (255, 255, 255, 255))
                mc_rgb = Image.alpha_composite(bg, mc_rgba).convert("RGB")
                crop = mc_rgb
            else:
                crop = image.crop((x0, y0, x1, y1)).convert("RGB")

            feat_path = feature_dir / f"box_{idx}.pt"

            with torch.no_grad():
                dino_in = dino_preprocess(crop).unsqueeze(0).to(dino_device)
                out = dino_model.forward_features(dino_in)
                if isinstance(out, dict):
                    tokens = (
                        out.get("x_norm_patchtokens")
                        or out.get("x_norm_patch_tokens")
                        or out.get("patch_tokens")
                        or out.get("tokens")
                    )
                    if tokens is None:
                        tokens = out.get("x_norm_clstoken") or out.get("x_norm_cls_token") or out.get("cls_token")
                    if tokens is None:
                        raise ValueError("DINO features not found")
                else:
                    tokens = out
                if tokens.dim() == 3:  # [B, N, D]
                    pooled = tokens.mean(dim=1)
                else:
                    pooled = tokens
                dino_feat = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                torch.save(dino_feat.cpu(), feat_path)

        print(f"保存先: {prompt_dir}")
        print(f"BBox保存: {bbox_path}")
        print(f"BBox特徴保存: {feature_dir} (count={len(boxes_padded)})")

    # det 画像と同じタイミングで、全ボックスの注釈画像を保存
    if annot_name:
        def trim_whitespace(img: Image.Image):
            arr = np.array(img)
            mask = np.any(arr != 255, axis=-1)
            coords = np.argwhere(mask)
            if coords.size == 0:
                return img
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            return img.crop((x0, y0, x1, y1))

        def load_boxes_all(base_dir: Path):
            boxes = []
            masks = []
            bbox_root = base_dir / "BoundingBox"
            for pdir in sorted(bbox_root.glob("*")):
                if not pdir.is_dir():
                    continue
                bbox_path = pdir / "boxes.json"
                mask_dir = pdir / "Masks"
                if not bbox_path.exists():
                    continue
                with open(bbox_path, "r") as f:
                    arr = json.load(f)
                for idx, b in enumerate(arr):
                    boxes.append({"box": b})
                    mask_path = mask_dir / f"mask_{idx}.png"
                    masks.append(mask_path if mask_path.exists() else None)
            return boxes, masks

        boxes_all, masks_all = load_boxes_all(base_dir)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        color = "red"

        def draw_mask_contour(mask_path):
            if mask_path is None:
                return False
            try:
                m = np.array(Image.open(mask_path).convert("L"))
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                drawn = False
                for c in contours:
                    c = c.squeeze()
                    if c.ndim != 2 or c.shape[0] < 2:
                        continue
                    ax.plot(c[:, 0], c[:, 1], color=color, linewidth=1.0)
                    drawn = True
                return drawn
            except Exception:
                return False

        for idx_box, item in enumerate(boxes_all):
            b = item["box"]
            drawn = draw_mask_contour(masks_all[idx_box])
            if not drawn:
                rect = plt.Rectangle(
                    (b[0], b[1]),
                    b[2] - b[0],
                    b[3] - b[1],
                    linewidth=1.0,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
            if draw_index:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                ax.text(
                    cx,
                    cy,
                    f"{idx_box}",
                    color=color,
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor=color, boxstyle="round,pad=0.1"),
                )
        ax.axis("off")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img_out = Image.fromarray(rgba[..., :3])
        img_out = trim_whitespace(img_out)
        out_path = base_dir / annot_name
        img_out.save(out_path)
        plt.close(fig)
        print(f"annot image saved: {out_path}")

    return base_dir, total_boxes


# def compute_relations_2d(base_dir: Path):
#     from itertools import combinations
#     import math

#     bbox_root = base_dir / "BoundingBox"
#     relation_dir = base_dir / "relation"
#     if relation_dir.exists():
#         import shutil
#         shutil.rmtree(relation_dir)
#     relation_dir.mkdir(parents=True, exist_ok=True)

#     boxes_all = []
#     for pdir in sorted(bbox_root.glob("*")):
#         if not pdir.is_dir():
#             continue
#         prompt_norm = pdir.name.replace("_", " ")
#         bbox_path = pdir / "boxes.json"
#         if not bbox_path.exists():
#             continue
#         with open(bbox_path, "r") as f:
#             boxes = json.load(f)
#         for b in boxes:
#             x0, y0, x1, y1 = b
#             cx = (x0 + x1) / 2.0
#             cy = (y0 + y1) / 2.0
#             boxes_all.append({"prompt": prompt_norm, "box": b, "center": (cx, cy)})

#     relations = []
#     for i, j in combinations(range(len(boxes_all)), 2):
#         a = boxes_all[i]
#         b = boxes_all[j]
#         dx = b["center"][0] - a["center"][0]
#         dy = b["center"][1] - a["center"][1]
#         dist = math.hypot(dx, dy)
#         ang_ab = math.degrees(math.atan2(dy, dx))  # A->B
#         ang_ba = math.degrees(math.atan2(-dy, -dx))  # B->A
#         relations.append(
#             {
#                 "from_idx": i,
#                 "to_idx": j,
#                 "from_prompt": a["prompt"],
#                 "to_prompt": b["prompt"],
#                 "distance": dist,
#                 "angle_deg": ang_ab,
#                 "vector": [dx, dy],
#             }
#         )
#         relations.append(
#             {
#                 "from_idx": j,
#                 "to_idx": i,
#                 "from_prompt": b["prompt"],
#                 "to_prompt": a["prompt"],
#                 "distance": dist,
#                 "angle_deg": ang_ba,
#                 "vector": [-dx, -dy],
#             }
#         )

#     rel_path = relation_dir / "relations.json"
#     with open(rel_path, "w") as f:
#         json.dump(relations, f, indent=2)
#     print(f"relation saved: {rel_path}")
#     print(f"num boxes: {len(boxes_all)}, num relations: {len(relations)}")
#     return rel_path


def _resolve_conda_sh(conda_sh: str):
    candidates = [
        conda_sh,
        "~/anaconda3/etc/profile.d/conda.sh",
        "~/miniconda3/etc/profile.d/conda.sh",
    ]
    for c in candidates:
        p = Path(os.path.expanduser(c))
        if p.exists():
            return p
    raise FileNotFoundError(f"conda.sh が見つかりません: {candidates}")


def run_conda_command(env_name: str, command: str, conda_sh: str, workdir: Path | None = None):
    """
    conda 環境を切り替えてコマンドを実行するヘルパー。
    command はシェルスクリプトとして与える。
    """
    conda_sh = _resolve_conda_sh(conda_sh)
    activate = f"source {conda_sh} && conda activate {env_name} && {command}"
    print(f"[conda:{env_name}] {command}")
    proc = subprocess.run(["bash", "-lc", activate], cwd=str(workdir) if workdir else None, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"conda env {env_name} のコマンドが失敗しました: {command}")
    if proc.stdout:
        print(proc.stdout)
    return proc


def run_wildcamera(image_path: str, output_dir: Path, conda_sh: str, cmd_template: str):
    """
    WildCamera を実行して焦点距離 (focal) を取得する。
    cmd_template には {image} プレースホルダを含める。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_image = str(Path(image_path).resolve())
    cmd = cmd_template.format(image=abs_image)
    proc = run_conda_command("wildcamera", cmd, conda_sh, workdir=output_dir)
    focal = None
    fx = fy = cx = cy = None
    for line in proc.stdout.splitlines():
        if "Est Focal" in line:
            try:
                focal = float(line.strip().split()[-1])
            except Exception:
                continue
        if "Est Intrinsics" in line:
            # 例: Est Intrinsics: fx=..., fy=..., cx=..., cy=...
            try:
                parts = line.strip().split()
                for p in parts:
                    if p.startswith("fx="):
                        fx = float(p.split("fx=")[1].strip(","))
                    elif p.startswith("fy="):
                        fy = float(p.split("fy=")[1].strip(","))
                    elif p.startswith("cx="):
                        cx = float(p.split("cx=")[1].strip(","))
                    elif p.startswith("cy="):
                        cy = float(p.split("cy=")[1].strip(","))
            except Exception:
                pass
    if focal is not None:
        print(f"WildCamera focal: {focal}")
        # 保存しておく
        focal_path = output_dir / "focal.txt"
        focal_json = output_dir / "focal.json"
        with open(focal_path, "w") as f:
            f.write(f"{focal}\n")
        # 画像サイズから簡易 cx,cy を付与
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                w, h = im.size
            fxw = fx if fx is not None else focal
            fyw = fy if fy is not None else focal
            cxw = cx if cx is not None else w / 2.0
            cyw = cy if cy is not None else h / 2.0
            with open(focal_json, "w") as f:
                json.dump({"focal": focal, "fx": fxw, "fy": fyw, "cx": cxw, "cy": cyw}, f, indent=2)
            focal = fxw  # 後段用に fx を優先
            fx, fy, cx, cy = fxw, fyw, cxw, cyw
        except Exception:
            pass
    else:
        print("警告: WildCamera から焦点距離を取得できませんでした。")
    return {"focal": focal, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def run_perspectivefields(image_path: str, output_dir: Path, conda_sh: str, cmd_template: str):
    """
    PerspectiveFields を conda 環境 perspective 上で実行し、外部パラメータ (extrinsic) を保存する。
    cmd_template には {image}, {output}, {extrinsic} プレースホルダを含める。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_path = output_dir / "extrinsic.json"
    fmt_kwargs = {"image": str(Path(image_path).resolve()), "output": output_dir, "extrinsic": extrinsic_path}
    cmd = cmd_template.format(**fmt_kwargs)
    # demo.py はカレントディレクトリに出力する想定があるので workdir=output_dir で実行
    proc = run_conda_command("perspective", cmd, conda_sh, workdir=output_dir)
    if extrinsic_path.exists():
        print(f"extrinsic saved: {extrinsic_path}")
        return extrinsic_path

    # demo.py がファイルを書かない場合は stdout をパースして暫定 extrinsic を生成
    import re

    roll = pitch = vfov = cx = cy = None
    for line in proc.stdout.splitlines():
        m = re.search(r"roll:\s*([-+]?[0-9]*\\.?[0-9]+)", line)
        if m:
            roll = float(m.group(1))
        m = re.search(r"pitch:\s*([-+]?[0-9]*\\.?[0-9]+)", line)
        if m:
            pitch = float(m.group(1))
        m = re.search(r"vfov:\s*([-+]?[0-9]*\\.?[0-9]+)", line)
        if m:
            vfov = float(m.group(1))
        m = re.search(r"cx:\s*([-+]?[0-9]*\\.?[0-9]+)", line)
        if m:
            cx = float(m.group(1))
        m = re.search(r"cy:\s*([-+]?[0-9]*\\.?[0-9]+)", line)
        if m:
            cy = float(m.group(1))
    if roll is not None and pitch is not None:
        yaw = 0.0  # 情報が無いので 0 と仮定
        # ZYX (yaw→pitch→roll) で回転行列を構成
        r2d = math.pi / 180.0
        cr, sr = math.cos(roll * r2d), math.sin(roll * r2d)
        cp, sp = math.cos(pitch * r2d), math.sin(pitch * r2d)
        cyw, syw = math.cos(yaw * r2d), math.sin(yaw * r2d)
        Rz = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        t = np.zeros(3, dtype=np.float32)
        meta = {"roll_deg": roll, "pitch_deg": pitch, "yaw_deg_assumed": yaw, "vfov_deg": vfov, "cx": cx, "cy": cy}
        with open(extrinsic_path, "w") as f:
            json.dump({"rotation": R.tolist(), "translation": t.tolist(), "meta": meta}, f, indent=2)
        print(f"extrinsic synthesized from stdout and saved: {extrinsic_path}")
        return extrinsic_path

    print(f"警告: PerspectiveFields の出力 extrinsic が見つかりませんでした (期待: {extrinsic_path})")
    return None


def run_metric3d(image_path: str, output_dir: Path, conda_sh: str, cmd_template: str, override_intrinsics: dict | None = None):
    """
    Metric3Dv2 を conda 環境 metric3d 上で実行し、深度と内部パラメータを保存する。
    cmd_template には {image}, {output}, {depth}, {intrinsic} プレースホルダを含める。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_path = output_dir / "depth.npy"
    intrinsic_path = output_dir / "intrinsics.json"
    fmt_kwargs = {"image": str(Path(image_path).resolve()), "output": output_dir, "depth": depth_path, "intrinsic": intrinsic_path}
    # 事前に内部パラメータを指定する場合は書き込んでおく
    if override_intrinsics is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(intrinsic_path, "w") as f_io:
            json.dump(override_intrinsics, f_io, indent=2)
    cmd = cmd_template.format(**fmt_kwargs)
    proc = run_conda_command("metric3d", cmd, conda_sh, workdir=output_dir)

    # demo.py が save-npy を使わない場合は workdir 内の *_depth.npy を拾う
    if not depth_path.exists():
        cand = list(output_dir.glob("*_depth.npy"))
        if len(cand) > 0:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            depth_path = cand[0]
            print(f"拾得: {depth_path}")
        else:
            print(f"警告: Metric3D の出力 depth が見つかりませんでした (期待: {depth_path})")
            depth_path = None

    if not intrinsic_path.exists():
        # demo では intrinsics を出さないため、画像サイズから単純モデルで合成
        try:
            with Image.open(image_path) as im:
                w, h = im.size
            f = max(w, h)  # 簡易仮定: fx=fy=max(W,H)
            intr = {"fx": float(f), "fy": float(f), "cx": float(w) / 2.0, "cy": float(h) / 2.0}
            # override があれば上書き
            if override_intrinsics is not None:
                intr["fx"] = float(override_intrinsics.get("fx", intr["fx"]))
                intr["fy"] = float(override_intrinsics.get("fy", intr["fy"]))
                intr["cx"] = float(override_intrinsics.get("cx", intr["cx"]))
                intr["cy"] = float(override_intrinsics.get("cy", intr["cy"]))
            with open(intrinsic_path, "w") as f_io:
                json.dump(intr, f_io, indent=2)
            print(f"intrinsics synthesized from image size and saved: {intrinsic_path}")
        except Exception as e:  # noqa: BLE001
            print(f"警告: Metric3D の出力 intrinsics が見つからず、合成にも失敗しました: {e}")
            intrinsic_path = None

    if depth_path is not None:
        print(f"depth saved: {depth_path}")
    if intrinsic_path is not None:
        print(f"intrinsics saved: {intrinsic_path}")

    # override_intrinsics がある場合はスケールを合わせて上書き
    if depth_path is not None and override_intrinsics is not None:
        try:
            d = np.load(depth_path)
            if d.ndim == 3:
                d = d.squeeze()
            try:
                intr_orig = load_intrinsics(intrinsic_path)
            except Exception:
                intr_orig = override_intrinsics
            scale = float(override_intrinsics["fx"]) / float(intr_orig.get("fx", override_intrinsics["fx"]))
            # cx, cy を画像サイズから補完
            if "cx" not in override_intrinsics or "cy" not in override_intrinsics:
                with Image.open(image_path) as im:
                    w, h = im.size
                override_intrinsics = {
                    "fx": float(override_intrinsics["fx"]),
                    "fy": float(override_intrinsics.get("fy", override_intrinsics["fx"])),
                    "cx": float(override_intrinsics.get("cx", w / 2.0)),
                    "cy": float(override_intrinsics.get("cy", h / 2.0)),
                }
            d = d * scale
            np.save(depth_path, d)
            with open(intrinsic_path, "w") as f_io:
                json.dump(override_intrinsics, f_io, indent=2)
            print(f"depth rescaled to wildcamera intrinsics (scale={scale:.3f}) and intrinsics overwritten.")
        except Exception as e:
            print(f"警告: override_intrinsics での深度スケール調整に失敗しました: {e}")

    # 追加のカラーマップ（plasma系）を保存
    if depth_path is not None:
        try:
            d = np.load(depth_path)
            if d.ndim == 3:
                d = d.squeeze()
            mask = np.isfinite(d) & (d > 0)
            if mask.any():
                dn = d.copy()
                vmin = np.percentile(dn[mask], 2.0)
                vmax = np.percentile(dn[mask], 98.0)
                dn = np.clip((dn - vmin) / (vmax - vmin + 1e-8), 0, 1)
            else:
                dn = np.zeros_like(d)
            plt.imshow(dn, cmap="magma")
            plt.axis("off")
            plt.tight_layout(pad=0)
            magma_path = output_dir / "depth_magma.png"
            plt.savefig(magma_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"depth colormap (magma) saved: {magma_path}")
        except Exception as e:  # noqa: BLE001
            print(f"警告: depth のカラーマップ生成に失敗しました: {e}")

    return depth_path, intrinsic_path


def load_intrinsics(intrinsic_path: Path):
    with open(intrinsic_path, "r") as f:
        data = json.load(f)
    required = ["fx", "fy", "cx", "cy"]
    for k in required:
        if k not in data:
            raise ValueError(f"intrinsics に {k} がありません: {intrinsic_path}")
    return data


def load_extrinsic(extrinsic_path: Path):
    with open(extrinsic_path, "r") as f:
        data = json.load(f)
    if "rotation" not in data or "translation" not in data:
        raise ValueError(f"extrinsic に rotation/translation がありません: {extrinsic_path}")
    R = np.asarray(data["rotation"], dtype=np.float32)
    t = np.asarray(data["translation"], dtype=np.float32).reshape(-1)
    if R.shape != (3, 3) or t.shape[0] != 3:
        raise ValueError(f"extrinsic 形状が不正です: R {R.shape}, t {t.shape}")
    return {"rotation": R, "translation": t}


def load_aabbs(base_dir: Path):
    relation_dir = base_dir / "relation"
    path_norm = relation_dir / "aabbs_world_norm.json"
    path_world = relation_dir / "aabbs_world.json"
    aabb_path = path_norm if path_norm.exists() else path_world
    if not aabb_path.exists():
        return None
    with open(aabb_path, "r") as f:
        data = json.load(f)
    return data


def load_point_clouds(base_dir: Path):
    relation_dir = base_dir / "relation" / "point_clouds"
    if not relation_dir.exists():
        return None
    clouds = {}
    # Prefer normalized point clouds if available
    for p in sorted(relation_dir.glob("box_*_norm.npy")):
        try:
            idx = int(p.stem.split("_")[1])
        except Exception:
            continue
        clouds[idx] = np.load(p)
    if clouds:
        return clouds
    for p in sorted(relation_dir.glob("box_*.npy")):
        try:
            idx = int(p.stem.split("_")[1])
        except Exception:
            continue
        clouds[idx] = np.load(p)
    return clouds if clouds else None


def compute_aabb_distance(aabb_a: dict, aabb_b: dict):
    aminv = np.array(aabb_a["min"], dtype=np.float32)
    amaxv = np.array(aabb_a["max"], dtype=np.float32)
    bminv = np.array(aabb_b["min"], dtype=np.float32)
    bmaxv = np.array(aabb_b["max"], dtype=np.float32)
    dx = max(0.0, bminv[0] - amaxv[0], aminv[0] - bmaxv[0])
    dy = max(0.0, bminv[1] - amaxv[1], aminv[1] - bmaxv[1])
    dz = max(0.0, bminv[2] - amaxv[2], aminv[2] - bmaxv[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def load_depth(depth_path: Path):
    depth = np.load(depth_path)
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth 配列の形状が不正です: {depth.shape}")
    return depth


def compute_relations_3d(base_dir: Path, depth: np.ndarray, intrinsics: dict, extrinsic: dict):
    """
    深度とカメラパラメータから3D座標を復元し、外部パラメータで正規化した上で relation を生成。
    - 各バウンディングボックス内の有効ピクセルを3D化し、boxごとの点群を保存
    - 全点群を正規化（重心原点シフト）した point_cloud_all.npy を保存
    - relation/relations.json に 3D ベクトルと画像平面上の角度 (angle_deg) を保存する
    """
    bbox_root = base_dir / "BoundingBox"
    relation_dir = base_dir / "relation"
    if relation_dir.exists():
        import shutil
        shutil.rmtree(relation_dir)
    relation_dir.mkdir(parents=True, exist_ok=True)

    boxes_all = []
    for pdir in sorted(bbox_root.glob("*")):
        if not pdir.is_dir():
            continue
        prompt_norm = pdir.name.replace("_", " ")
        bbox_path = pdir / "boxes.json"
        mask_dir = pdir / "Masks"
        if not bbox_path.exists():
            continue
        with open(bbox_path, "r") as f:
            boxes = json.load(f)
        for idx_box, b in enumerate(boxes):
            x0, y0, x1, y1 = b
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            mpath = mask_dir / f"mask_{idx_box}.png"
            boxes_all.append({"prompt": prompt_norm, "box": b, "center": (cx, cy), "mask_path": mpath if mpath.exists() else None})

    if len(boxes_all) == 0:
        raise RuntimeError("BoundingBox が存在しません。")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx0 = float(intrinsics.get("cx", depth.shape[1] / 2))
    cy0 = float(intrinsics.get("cy", depth.shape[0] / 2))
    R = extrinsic["rotation"]
    t = extrinsic["translation"]

    def pixel_to_world(px, py, d):
        x_cam = (px - cx0) / fx * d
        y_cam = (py - cy0) / fy * d
        cam = np.array([x_cam, y_cam, d], dtype=np.float32)
        world = R @ cam + t
        return world

    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    default_depth = float(np.median(valid_depth)) if valid_depth.size > 0 else None
    if default_depth is None:
        raise RuntimeError("深度が有効ではありません。")

    points_world_center = []
    per_box_points = []
    pc_dir = relation_dir / "point_clouds"
    pc_dir.mkdir(parents=True, exist_ok=True)

    aabbs_world = []
    aabbs_world_norm = []

    for idx, item in enumerate(boxes_all):
        x0, y0, x1, y1 = item["box"]
        xs = slice(int(max(0, math.floor(y0))), int(min(depth.shape[0], math.ceil(y1))))
        ys = slice(int(max(0, math.floor(x0))), int(min(depth.shape[1], math.ceil(x1))))
        region = depth[xs, ys]
        H = region.shape[0]
        W = region.shape[1]
        yy, xx = np.meshgrid(
            np.arange(xs.start, xs.start + H, dtype=np.float32),
            np.arange(ys.start, ys.start + W, dtype=np.float32),
            indexing="ij",
        )
        depth_valid = np.isfinite(region) & (region > 0)
        # マスクがあれば物体領域のみを使用
        mpath = item.get("mask_path")
        if mpath is not None and mpath.exists():
            try:
                m_full = np.array(Image.open(mpath).convert("L"))
                m_slice = m_full[xs, ys] > 0
                depth_valid = depth_valid & m_slice
            except Exception:
                pass
        if not depth_valid.any():
            # depth が無い場合は中央値にフォールバック
            d_val = default_depth
            cx_p, cy_p = item["center"]
            world = pixel_to_world(cx_p, cy_p, d_val)
            item["depth"] = d_val
            item["world"] = world
            points_world_center.append(world)
            per_box_points.append(np.empty((0, 3), dtype=np.float32))
            np.save(pc_dir / f"box_{idx}.npy", per_box_points[-1])
            continue

        z = region[depth_valid]
        x_pix = xx[depth_valid]
        y_pix = yy[depth_valid]
        x_cam = (x_pix - cx0) / fx * z
        y_cam = (y_pix - cy0) / fy * z
        cams = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
        worlds = (R @ cams.T + t.reshape(3, 1)).T  # [N,3]
        per_box_points.append(worlds)
        np.save(pc_dir / f"box_{idx}.npy", worlds)
        world_center = np.median(worlds, axis=0).astype(np.float32)
        aabbs_world.append(
            {
                "min": worlds.min(axis=0).tolist(),
                "max": worlds.max(axis=0).tolist(),
                "center": world_center.tolist(),
                "prompt": item["prompt"],
                "box_idx": idx,
            }
        )

        # 中心代表点（点群中央値）を保持（後続のrelation計算用）
        item["depth"] = float(np.median(z))
        item["world"] = world_center
        points_world_center.append(world_center)

    points_world_center = np.stack(points_world_center, axis=0)
    # シーン正規化: 重心を原点に平行移動（全点群の結合で計算）
    if per_box_points:
        all_pts = np.concatenate(per_box_points, axis=0) if any(p.size > 0 for p in per_box_points) else points_world_center
    else:
        all_pts = points_world_center
    centroid = all_pts.mean(axis=0, keepdims=True)
    points_world_center = points_world_center - centroid
    per_box_points_norm = [p - centroid for p in per_box_points]
    np.save(relation_dir / "point_cloud_all.npy", all_pts - centroid)
    for i, p in enumerate(per_box_points_norm):
        np.save(pc_dir / f"box_{i}_norm.npy", p)
        if p.size > 0:
            aabbs_world_norm.append(
                {
                    "min": p.min(axis=0).tolist(),
                    "max": p.max(axis=0).tolist(),
                    "prompt": boxes_all[i]["prompt"],
                    "box_idx": i,
                }
            )
    for i, item in enumerate(boxes_all):
        item["world"] = points_world_center[i]

    from itertools import combinations
    relations = []
    for i, j in combinations(range(len(boxes_all)), 2):
        a = boxes_all[i]
        b = boxes_all[j]
        vec3 = b["world"] - a["world"]
        dist3 = float(np.linalg.norm(vec3))
        dx_img = b["center"][0] - a["center"][0]
        dy_img = b["center"][1] - a["center"][1]
        dist_img = math.hypot(dx_img, dy_img)
        ang_img = math.degrees(math.atan2(dy_img, dx_img))

        rel_ab = {
            "from_idx": i,
            "to_idx": j,
            "from_prompt": a["prompt"],
            "to_prompt": b["prompt"],
            "distance": dist3,
            "angle_deg": ang_img,  # image-plane angle (互換性のため)
            "vector": vec3.tolist(),
            "distance_image_px": dist_img,
            "vector_image": [dx_img, dy_img],
        }
        rel_ba = {
            "from_idx": j,
            "to_idx": i,
            "from_prompt": b["prompt"],
            "to_prompt": a["prompt"],
            "distance": dist3,
            "angle_deg": math.degrees(math.atan2(-dy_img, -dx_img)),
            "vector": (-vec3).tolist(),
            "distance_image_px": dist_img,
            "vector_image": [-dx_img, -dy_img],
        }
        relations.append(rel_ab)
        relations.append(rel_ba)

    rel_path = relation_dir / "relations.json"
    with open(rel_path, "w") as f:
        json.dump(relations, f, indent=2)
    # 代表点（ボックス中心ベース）の正規化点群も保存
    np.save(relation_dir / "point_cloud_centers.npy", points_world_center)
    # AABB を保存
    with open(relation_dir / "aabbs_world.json", "w") as f:
        json.dump(aabbs_world, f, indent=2)
    with open(relation_dir / "aabbs_world_norm.json", "w") as f:
        json.dump(aabbs_world_norm, f, indent=2)
    # AABB 重心をテキストで保存（ワールド座標系、点群中央値）
    aabb_centers_path = relation_dir / "aabb_centers.txt"
    with open(aabb_centers_path, "w") as f:
        for aabb in aabbs_world:
            center = np.array(aabb.get("center", (0.0, 0.0, 0.0)), dtype=float)
            idx = aabb.get("box_idx", -1)
            f.write(f"パーツ {idx}: x={center[0]:.6f}, y={center[1]:.6f}, z={center[2]:.6f}\n")
    print(f"3D relation saved: {rel_path}")
    return rel_path


def preprocess_image_square_for_vggt(image_path: str, target_size: int = 518):
    from torchvision import transforms as TF
    img = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    width, height = img.size
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square_img.paste(img, (left, top))
    square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    img_tensor = TF.ToTensor()(square_img)
    info = {
        "orig_w": width,
        "orig_h": height,
        "max_dim": max_dim,
        "left": left,
        "top": top,
        "target_size": target_size,
    }
    return img_tensor, info


def transform_mask_to_pointmap(mask_path: Path, info: dict):
    max_dim = int(info["max_dim"])
    left = int(info["left"])
    top = int(info["top"])
    target_size = int(info["target_size"])
    m = ImageOps.exif_transpose(Image.open(mask_path).convert("L"))
    square = Image.new("L", (max_dim, max_dim), 0)
    square.paste(m, (left, top))
    square = square.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.array(square) > 0


def run_vggt_pointmap(image_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    point_path = output_dir / "world_points.npy"
    meta_path = output_dir / "preprocess.json"
    if point_path.exists() and meta_path.exists():
        with open(meta_path, "r") as f:
            info = json.load(f)
        return np.load(point_path), info

    vggt_root = Path("/home/yu/vggt")
    if vggt_root.exists():
        sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    img_tensor, info = preprocess_image_square_for_vggt(image_path, target_size=518)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    def _run_vggt(run_device: str, run_dtype: torch.dtype):
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(run_device)
        model.eval()
        images = img_tensor.to(run_device, dtype=run_dtype)
        with torch.no_grad():
            if run_device == "cuda":
                with torch.cuda.amp.autocast(dtype=run_dtype):
                    preds = model(images)
            else:
                preds = model(images)
        return preds

    try:
        predictions = _run_vggt(device, dtype)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            print("[warn] VGGT CUDA OOM; retrying on CPU")
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            predictions = _run_vggt("cpu", torch.float32)
        else:
            raise

    # Use VGGT camera parameters to unproject depth to 3D
    pose_enc = predictions["pose_enc"]
    depth = predictions["depth"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, img_tensor.shape[-2:])
    # Strip batch dimension if present (VGGT returns BxSx...)
    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic is not None and intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    pts = world_points[0].astype(np.float32)
    np.save(point_path, pts)
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)
    return pts, info


def compute_relations_from_pointmap(base_dir: Path, point_map: np.ndarray, info: dict):
    """
    VGGT の world_points (H,W,3) とマスクから点群を生成し、relation を構築する。
    """
    bbox_root = base_dir / "BoundingBox"
    relation_dir = base_dir / "relation"
    if relation_dir.exists():
        import shutil
        shutil.rmtree(relation_dir)
    relation_dir.mkdir(parents=True, exist_ok=True)

    boxes_all = []
    for pdir in sorted(bbox_root.glob("*")):
        if not pdir.is_dir():
            continue
        prompt_norm = pdir.name.replace("_", " ")
        bbox_path = pdir / "boxes.json"
        mask_dir = pdir / "Masks"
        if not bbox_path.exists():
            continue
        with open(bbox_path, "r") as f:
            boxes = json.load(f)
        for idx_box, b in enumerate(boxes):
            x0, y0, x1, y1 = b
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            mpath = mask_dir / f"mask_{idx_box}.png"
            boxes_all.append({"prompt": prompt_norm, "box": b, "center": (cx, cy), "mask_path": mpath if mpath.exists() else None})

    if len(boxes_all) == 0:
        raise RuntimeError("BoundingBox が存在しません。")

    max_dim = float(info["max_dim"])
    left = float(info["left"])
    top = float(info["top"])
    target_size = float(info["target_size"])
    scale = target_size / max_dim if max_dim > 0 else 1.0

    points_world_center = []
    per_box_points = []
    pc_dir = relation_dir / "point_clouds"
    pc_dir.mkdir(parents=True, exist_ok=True)

    aabbs_world = []
    aabbs_world_norm = []

    for idx, item in enumerate(boxes_all):
        mpath = item.get("mask_path")
        mask_map = None
        if mpath is not None and mpath.exists():
            try:
                mask_map = transform_mask_to_pointmap(mpath, info)
            except Exception:
                mask_map = None

        if mask_map is not None:
            pts = point_map[mask_map]
            pts = pts[np.all(np.isfinite(pts), axis=1)]
        else:
            pts = np.empty((0, 3), dtype=np.float32)

        if pts.size == 0:
            cx_p, cy_p = item["center"]
            px = int(round((cx_p + left) * scale))
            py = int(round((cy_p + top) * scale))
            px = max(0, min(point_map.shape[1] - 1, px))
            py = max(0, min(point_map.shape[0] - 1, py))
            world_center = point_map[py, px].astype(np.float32)
            points_world_center.append(world_center)
            per_box_points.append(np.empty((0, 3), dtype=np.float32))
            np.save(pc_dir / f"box_{idx}.npy", per_box_points[-1])
            continue

        per_box_points.append(pts.astype(np.float32))
        np.save(pc_dir / f"box_{idx}.npy", per_box_points[-1])
        world_center = np.median(pts, axis=0).astype(np.float32)
        aabbs_world.append(
            {
                "min": pts.min(axis=0).tolist(),
                "max": pts.max(axis=0).tolist(),
                "center": world_center.tolist(),
                "prompt": item["prompt"],
                "box_idx": idx,
            }
        )
        points_world_center.append(world_center)

    points_world_center = np.stack(points_world_center, axis=0)
    if per_box_points:
        all_pts = np.concatenate(per_box_points, axis=0) if any(p.size > 0 for p in per_box_points) else points_world_center
    else:
        all_pts = points_world_center
    centroid = all_pts.mean(axis=0, keepdims=True)
    points_world_center = points_world_center - centroid
    per_box_points_norm = [p - centroid for p in per_box_points]
    np.save(relation_dir / "point_cloud_all.npy", all_pts - centroid)
    for i, p in enumerate(per_box_points_norm):
        np.save(pc_dir / f"box_{i}_norm.npy", p)
        if p.size > 0:
            aabbs_world_norm.append(
                {
                    "min": p.min(axis=0).tolist(),
                    "max": p.max(axis=0).tolist(),
                    "prompt": boxes_all[i]["prompt"],
                    "box_idx": i,
                }
            )
    for i, item in enumerate(boxes_all):
        item["world"] = points_world_center[i]

    from itertools import combinations
    relations = []
    for i, j in combinations(range(len(boxes_all)), 2):
        a = boxes_all[i]
        b = boxes_all[j]
        vec3 = b["world"] - a["world"]
        dist3 = float(np.linalg.norm(vec3))
        dx_img = b["center"][0] - a["center"][0]
        dy_img = b["center"][1] - a["center"][1]
        dist_img = math.hypot(dx_img, dy_img)
        ang_img = math.degrees(math.atan2(dy_img, dx_img))

        rel_ab = {
            "from_idx": i,
            "to_idx": j,
            "from_prompt": a["prompt"],
            "to_prompt": b["prompt"],
            "distance": dist3,
            "angle_deg": ang_img,
            "vector": vec3.tolist(),
            "distance_image_px": dist_img,
            "vector_image": [dx_img, dy_img],
        }
        rel_ba = {
            "from_idx": j,
            "to_idx": i,
            "from_prompt": b["prompt"],
            "to_prompt": a["prompt"],
            "distance": dist3,
            "angle_deg": math.degrees(math.atan2(-dy_img, -dx_img)),
            "vector": (-vec3).tolist(),
            "distance_image_px": dist_img,
            "vector_image": [-dx_img, -dy_img],
        }
        relations.append(rel_ab)
        relations.append(rel_ba)

    rel_path = relation_dir / "relations.json"
    with open(rel_path, "w") as f:
        json.dump(relations, f, indent=2)
    np.save(relation_dir / "point_cloud_centers.npy", points_world_center)
    with open(relation_dir / "aabbs_world.json", "w") as f:
        json.dump(aabbs_world, f, indent=2)
    with open(relation_dir / "aabbs_world_norm.json", "w") as f:
        json.dump(aabbs_world_norm, f, indent=2)
    aabb_centers_path = relation_dir / "aabb_centers.txt"
    with open(aabb_centers_path, "w") as f:
        for aabb in aabbs_world:
            center = np.array(aabb.get("center", (0.0, 0.0, 0.0)), dtype=float)
            idx = aabb.get("box_idx", -1)
            f.write(f"パーツ {idx}: x={center[0]:.6f}, y={center[1]:.6f}, z={center[2]:.6f}\n")
    print(f"3D relation saved: {rel_path}")
    return rel_path

def compute_relations(
    base_dir: Path,
    depth: np.ndarray | None = None,
    intrinsics: dict | None = None,
    extrinsic: dict | None = None,
    depth_path: Path | None = None,
    intrinsics_path: Path | None = None,
    extrinsic_path: Path | None = None,
):
    """
    3D 前提で relation を計算する。必須: depth, intrinsics, extrinsic。
    いずれか欠ける場合は例外を投げる。
    """
    if depth is None and depth_path is not None:
        depth = load_depth(depth_path)
    if intrinsics is None and intrinsics_path is not None:
        intrinsics = load_intrinsics(intrinsics_path)
    if extrinsic is None and extrinsic_path is not None:
        extrinsic = load_extrinsic(extrinsic_path)

    if depth is None or intrinsics is None or extrinsic is None:
        raise RuntimeError("3D relation を計算するには depth / intrinsics / extrinsic が必要です。")

    return compute_relations_3d(base_dir, depth, intrinsics, extrinsic)


def load_all_feat_vectors(base_dir: Path, allow_prompts=None):
    vecs = []
    labels = []
    allow_set = None if allow_prompts is None else set(allow_prompts)
    bbox_root = base_dir / "BoundingBox"
    for pdir in sorted(bbox_root.glob("*")):
        if not pdir.is_dir():
            continue
        prompt_dir = pdir.name
        prompt_norm = prompt_dir.replace("_", " ")
        if allow_set is not None and prompt_norm not in allow_set:
            continue
        feat_dir = pdir / "BBox_Feature"
        if not feat_dir.exists():
            continue
        for fpath in sorted(feat_dir.glob("box_*.pt")):
            f = torch.load(fpath)
            if f.dim() == 4:
                f = f.squeeze(0)
            if f.dim() == 3:
                v = f.mean(dim=(1, 2))
            elif f.dim() == 2:
                v = f.squeeze(0)
            elif f.dim() == 1:
                v = f
            else:
                continue
            v = torch.nn.functional.normalize(v, p=2, dim=0)
            vecs.append(v)
            labels.append(f"{prompt_norm}:{fpath.stem}")
    return (torch.stack(vecs) if vecs else torch.empty((0,)), labels)


def angle_to_dir(angle):
    # 画像座標(y下向き)に合わせて反転して判定
    ang = (-angle) % 360
    if (315 <= ang < 360) or (0 <= ang < 45):
        return "right"
    elif 45 <= ang < 135:
        return "above"
    elif 135 <= ang < 225:
        return "left"
    elif 225 <= ang < 315:
        return "under"
    return "unknown"


def energy_search(vecs0, vecs1, labels0, labels1, base1, manual_path, run_dir, contact_eps=0.01, alpha=2.0, beta=1.0):
    sim_mat = torch.matmul(vecs0, vecs1.t()).cpu().numpy()
    N0, N1 = sim_mat.shape

    beta_use = beta
    rel_data = []
    aabbs = []
    if beta_use != 0.0:
        relation_path = base1 / "relation" / "relations.json"
        if not relation_path.exists():
            print(f"SCIP LP: relation file not found, geometry disabled: {relation_path}")
            beta_use = 0.0
        else:
            with open(relation_path, "r") as f:
                rel_data = json.load(f)
            aabbs = load_aabbs(base1)
    contact_cache = {}
    # 外観／幾何の重み（LP も総当たりも同じ値を使用）

    # SCIP による線形割当てを試す（成功しない場合は従来探索にフォールバック）
    use_scip = False
    try:
        from pyscipopt import Model, quicksum  # type: ignore

        use_scip = True
    except Exception:
        use_scip = False
    # LP は 1 対 1 かつ正方行列を想定
    def try_scip_lp():
        if not use_scip:
            print("SCIP LP: PySCIPOpt が利用できないためスキップ")
            return None
        if N0 != N1:
            print(f"SCIP LP: N0({N0}) != N1({N1}) のためスキップ")
            return None
        # y 座標は AABB 重心から取得（無ければ point_cloud_centers.npy をフォールバック）
        ys = [None] * N1
        if beta_use != 0.0:
            if aabbs:
                for aabb in aabbs:
                    idx = aabb.get("box_idx")
                    if idx is None or idx >= N1:
                        continue
                    if "center" in aabb:
                        center = np.array(aabb["center"], dtype=float)
                    else:
                        mn = np.array(aabb["min"], dtype=float)
                        mx = np.array(aabb["max"], dtype=float)
                        center = (mn + mx) / 2.0
                    ys[idx] = float(center[1])
            # フォールバック: AABB から埋まらなかったものは point_cloud_centers を使用
            if any(y is None for y in ys):
                pc_cent_path = base1 / "relation" / "point_cloud_centers.npy"
                if not pc_cent_path.exists():
                    print(f"SCIP LP: {pc_cent_path} が存在しないためスキップ")
                    return None
                coords = np.load(pc_cent_path)
                if coords.shape[0] != N1 or coords.shape[1] < 3:
                    print(f"SCIP LP: point_cloud_centers 形状不正 {coords.shape}, N1={N1} のためスキップ")
                    return None
                for i in range(N1):
                    if ys[i] is None:
                        ys[i] = float(coords[i, 1])
            # 最終確認
            if any(y is None for y in ys):
                print("SCIP LP: y 座標が揃わないためスキップ")
                return None
        # AABB 距離ルックアップ
        aabb_distance_lookup = {}
        if beta_use != 0.0 and aabbs:
            for i in range(len(aabbs)):
                for j in range(i + 1, len(aabbs)):
                    ai = aabbs[i]["box_idx"]
                    aj = aabbs[j]["box_idx"]
                    d = compute_aabb_distance(aabbs[i], aabbs[j])
                    aabb_distance_lookup[(min(ai, aj), max(ai, aj))] = d
            # gap が 0 (しきい値内) の組だけ保存（base1 配下）
            dist_path = base1 / "aabb_distances.txt"
            with open(dist_path, "w") as f:
                for (ai, aj) in sorted(aabb_distance_lookup.keys()):
                    dval = aabb_distance_lookup[(ai, aj)]
                    if dval <= contact_eps:
                        f.write(f"{ai},{aj}: 0.000 (raw={dval:.6f})\n")
            print(f"aabb distances saved: {dist_path}")

        def aabb_distance_fn(j, l):
            return aabb_distance_lookup.get((min(j, l), max(j, l)), 0.0)

        # 外観エネルギー
        E_app = [[-float(sim_mat[i, j]) for j in range(N1)] for i in range(N0)]

        # relation をパース（above/under/contact のみ利用）
        rel_list = []
        if beta_use != 0.0:
            for line in manual_lines:
                try:
                    prefix, direction = line.split(":")
                    direction = direction.strip()
                    if "from" in prefix and "to" in prefix:
                        _, src, _, dst = prefix.split()
                        src = int(src)
                        dst = int(dst)
                    elif "and" in prefix:
                        a, _, b = prefix.split()
                        src = int(a)
                        dst = int(b)
                    elif "to" in prefix:
                        # 例: "0 to 1"
                        a, _, b = prefix.split()
                        src = int(a)
                        dst = int(b)
                    else:
                        continue
                except Exception:
                    continue
                if direction not in ("above", "under", "contact"):
                    continue
                if src >= N0 or dst >= N0:
                    continue
                rel_list.append((src, dst, direction))

        # above/under の連鎖を復元（manual自体は変更しない）
        if rel_list:
            above_edges = {(a, b) for (a, b, r) in rel_list if r == "above"}
            under_edges = {(a, b) for (a, b, r) in rel_list if r == "under"}
            # under は above の逆として統一
            for a, b in under_edges:
                above_edges.add((b, a))

            def transitive_closure(edges: set[tuple[int, int]]):
                closure = set(edges)
                for src in range(N0):
                    # BFS/DFS で到達可能なノードを列挙
                    stack = [dst for (s, dst) in closure if s == src]
                    seen = set(stack)
                    while stack:
                        cur = stack.pop()
                        for (s, dst) in list(closure):
                            if s != cur or dst in seen:
                                continue
                            closure.add((src, dst))
                            seen.add(dst)
                            stack.append(dst)
                return closure

            above_closure = transitive_closure(above_edges)
            # under は above の逆として復元
            under_closure = {(b, a) for (a, b) in above_closure}

            rel_set = set(rel_list)
            for a, b in above_closure:
                if a != b:
                    rel_set.add((a, b, "above"))
            for a, b in under_closure:
                if a != b:
                    rel_set.add((a, b, "under"))
            rel_list = list(rel_set)

        margin = 0.0
        def phi(r, j, l):
            # r は「j から見て l がどうあるべきか」を判定する
            if r == "above":
                # l が j より上（ys[l] < ys[j]）なら差分は正になりペナルティ 0
                dy = ys[j] - ys[l]
                return max(0.0, margin - dy)
            if r == "under":
                # l が j より下（ys[l] > ys[j]）なら差分は正になりペナルティ 0
                dy = ys[l] - ys[j]
                return max(0.0, margin - dy)
            if r == "contact":
                return max(0.0, aabb_distance_fn(j, l) - contact_eps)
            return 0.0

        model = Model("lp_match")
        # Stop SCIP if it exceeds 100 seconds
        try:
            model.setParam("limits/time", 100.0)
        except Exception:
            pass
        x = [[model.addVar(vtype="B", name=f"x_{i}_{j}") for j in range(N1)] for i in range(N0)]
        for i in range(N0):
            model.addCons(quicksum(x[i][j] for j in range(N1)) == 1)
        for j in range(N1):
            model.addCons(quicksum(x[i][j] for i in range(N0)) == 1)
        # relation ごとの制約:
        # - contact: 完成品側が非接触なら同時に 1 にできない
        use_contact_hard_constraints = True
        if beta_use != 0.0 and use_contact_hard_constraints:
            for (src, dst, r) in rel_list:
                if r != "contact":
                    continue
                # contact の場合のみ適用
                for j in range(N1):
                    for l in range(N1):
                        if j == l:
                            continue
                        if aabb_distance_fn(j, l) > contact_eps:
                            model.addCons(x[src][j] + x[dst][l] <= 1.0)

        obj_terms = []
        for i in range(N0):
            for j in range(N1):
                obj_terms.append(alpha * E_app[i][j] * x[i][j])

        # 幾何項（E_geo_ij を事前計算し、x_ij に載せる）
        E_geo = [[0.0 for _ in range(N1)] for _ in range(N0)]
        if beta_use != 0.0 and N1 > 1:
            p_uniform = 1.0 / float(N1 - 1)
            for i_rel in range(N0):
                for (src, dst, r) in rel_list:
                    if r == "contact":
                        if src != i_rel and dst != i_rel:
                            continue
                    else:
                        if src != i_rel:
                            continue
                    for j in range(N1):
                        for l in range(N1):
                            if j == l:
                                continue
                            phi_val = phi(r, j, l)
                            if phi_val != 0.0:
                                E_geo[i_rel][j] += phi_val * p_uniform
            for i in range(N0):
                for j in range(N1):
                    if E_geo[i][j] != 0.0:
                        obj_terms.append(beta * E_geo[i][j] * x[i][j])

        model.setObjective(quicksum(obj_terms), "minimize")
        model.optimize()
        status = model.getStatus()
        if status not in ("optimal", "bestsollimit"):
            print(f"SCIP LP: status={status} のためフォールバック")
            return None
        x_sol = [[model.getVal(x[i][j]) for j in range(N1)] for i in range(N0)]
        best_pairs = []
        for i in range(N0):
            j_best = max(range(N1), key=lambda j: x_sol[i][j])
            best_pairs.append(f"({i},{j_best})")
        best_energy_lp = float(model.getObjVal())
        # E_geo は事前計算済み（E_geo_ij）
        return {
            "pairs": best_pairs,
            "x": x_sol,
            "E_app": E_app,
            "E_geo": E_geo,
            "obj": best_energy_lp,
        }

    manual_lines = []
    if beta_use != 0.0:
        if manual_path is None:
            print("manual.txt not provided; geometry disabled.")
            beta_use = 0.0
        else:
            manual_path = Path(manual_path)
            if not manual_path.exists():
                print(f"manual.txt not found; geometry disabled: {manual_path}")
                beta_use = 0.0
            else:
                manual_lines = [ln.strip() for ln in manual_path.read_text().splitlines() if ln.strip()]

    if N0 == 0 or N1 == 0:
        raise RuntimeError(f"特徴ベクトルが空です: N0={N0}, N1={N1}")

    if N0 != N1:
        print(f"マッチング数不一致のため中止: N0={N0}, N1={N1}")
        return None
    # 小さい方のボックス数を基準にマッチング
    N_match = min(N0, N1)
    print(f"マッチング数: {N_match} (image0: {N0}, image1: {N1})")

    best_energy = float("inf")
    best_match = None
    all_results = []

    # まず SCIP の線形計画で解ける場合はそちらを優先（失敗時は従来探索へ）
    lp_result = try_scip_lp()
    if lp_result is not None:
        best_match = lp_result["pairs"]
        best_energy = lp_result["obj"]
        def _parse_pair_str(s):
            s = s.strip("() ")
            a, b = s.split(",")
            return int(a), int(b)
        pairs = [_parse_pair_str(s) for s in best_match]
        e_app_sum = 0.0
        e_geo_sum = 0.0
        e_app_mat = lp_result.get("E_app")
        e_geo_mat = lp_result.get("E_geo")
        if e_app_mat is not None:
            for i, j in pairs:
                if i < len(e_app_mat) and j < len(e_app_mat[i]):
                    e_app_sum += float(alpha) * float(e_app_mat[i][j])
        if e_geo_mat is not None:
            for i, j in pairs:
                if i < len(e_geo_mat) and j < len(e_geo_mat[i]):
                    e_geo_sum += float(beta_use) * float(e_geo_mat[i][j])
        e_total_str = f"{e_app_sum:.2f}+{e_geo_sum:.2f}"
        print(f"E_total (E^app+E^geo): {e_total_str}")
        match_dir = run_dir / "matching"
        match_dir.mkdir(parents=True, exist_ok=True)
        match_path = match_dir / "best_match.json"
        score_path = match_dir / "score_matrix.txt"
        e_total_path = match_dir / "E_total_matrix.txt"
        # コサイン類似度行列を保存（パーツ番号明記・小数3桁）
        with open(score_path, "w") as f:
            f.write("image0\\image1 " + " ".join(f"{j:>6d}" for j in range(N1)) + "\n")
            for i in range(N0):
                row_vals = " ".join(f"{sim_mat[i, j]:.3f}" for j in range(N1))
                f.write(f"パーツ {i}: {row_vals}\n")
        if e_app_mat is not None and e_geo_mat is not None:
            with open(e_total_path, "w") as f:
                f.write("image0\\image1 " + " ".join(f"{j:>10d}" for j in range(N1)) + "\n")
                for i in range(N0):
                    row_vals = " ".join(
                        f"{(float(alpha) * float(e_app_mat[i][j])):.2f}+{(float(beta_use) * float(e_geo_mat[i][j])):.2f}"
                        for j in range(N1)
                    )
                    f.write(f"パーツ {i}: {row_vals}\n")
        with open(match_path, "w") as f:
            json.dump(
                {
                    "best_pairs": best_match,
                    "E_total": float(best_energy),
                    "E_app": lp_result.get("E_app"),
                    "E_geo": lp_result.get("E_geo"),
                    "E_app_sum": e_app_sum,
                    "E_geo_sum": e_geo_sum,
                    "E_total_decomposed": e_total_str,
                    "alpha": float(alpha),
                    "beta": float(beta_use),
                    "solver": "scip_lp",
                },
                f,
                indent=2,
            )
        print(f"match result saved: {match_path}")
        print(f"score matrix saved: {score_path}")
        if e_app_mat is not None and e_geo_mat is not None:
            print(f"E_total matrix saved: {e_total_path}")
        return best_match, match_dir

    return None




def visualize_match(best_match, base0, base1, image0_path, image1_path, match_dir, vis_style="pair_color", show_index=False):
    if best_match is None or len(best_match) == 0:
        print("可視化対象のマッチングがありません。")
        return
    img0 = ImageOps.exif_transpose(Image.open(image0_path).convert("RGB"))
    img1 = ImageOps.exif_transpose(Image.open(image1_path).convert("RGB"))
    if vis_style == "pair_color":
        img0 = ImageOps.grayscale(img0).convert("RGB")
        img1 = ImageOps.grayscale(img1).convert("RGB")
    w0, h0 = img0.size
    w1, h1 = img1.size
    orig_w0, orig_h0 = w0, h0
    orig_w1, orig_h1 = w1, h1
    # 縦サイズを揃える（アスペクト比維持でリサイズ）
    target_h = max(h0, h1)
    scale0 = target_h / h0 if h0 != 0 else 1.0
    scale1 = target_h / h1 if h1 != 0 else 1.0
    if h0 != target_h:
        new_w0 = max(1, int(round(w0 * scale0)))
        img0 = img0.resize((new_w0, target_h), Image.LANCZOS)
        w0, h0 = img0.size
    if h1 != target_h:
        new_w1 = max(1, int(round(w1 * scale1)))
        img1 = img1.resize((new_w1, target_h), Image.LANCZOS)
        w1, h1 = img1.size

    def load_boxes_all(base_dir: Path):
        boxes = []
        masks = []
        id_arr = None
        id_npy = base_dir / "IdMask" / "id_mask.npy"
        if id_npy.exists():
            try:
                id_arr = np.load(id_npy)
            except Exception:
                id_arr = None
        bbox_root = base_dir / "BoundingBox"
        for pdir in sorted(bbox_root.glob("*")):
            if not pdir.is_dir():
                continue
            prompt_norm = pdir.name.replace("_", " ")
            bbox_path = pdir / "boxes.json"
            mask_dir = pdir / "Masks"
            if not bbox_path.exists():
                continue
            with open(bbox_path, "r") as f:
                arr = json.load(f)
            for idx, b in enumerate(arr):
                boxes.append({"prompt": prompt_norm, "box": b})
                mask_path = mask_dir / f"mask_{idx}.png"
                masks.append(mask_path if mask_path.exists() else None)
        return boxes, masks, id_arr

    boxes0, masks0, id_arr0 = load_boxes_all(base0)
    boxes1, masks1, id_arr1 = load_boxes_all(base1)

    def load_answer_pairs(image1_path: Path, match_dir: Path):
        import re
        candidates = [
            Path("/home/yu/Otani/answer") / f"{image1_path.stem}.txt",
            image1_path.parent / f"answer_{image1_path.stem}.txt",
            image1_path.parent / "answer.txt",
            match_dir / "answer.txt",
        ]
        ans_path = next((p for p in candidates if p.exists()), None)
        if ans_path is None:
            return None
        pairs = set()
        with open(ans_path, "r") as f:
            for line in f:
                m = re.search(r"\((\d+)\s*,\s*(\d+)\)", line)
                if not m:
                    continue
                pairs.add((int(m.group(1)), int(m.group(2))))
        return pairs

    def parse_pair_str(s):
        s = s.strip("() ")
        a, b = s.split(",")
        return int(a), int(b)

    pairs = [parse_pair_str(s) for s in best_match]
    answer_pairs = load_answer_pairs(Path(image1_path), Path(match_dir))

    def trim_whitespace(img: Image.Image):
        arr = np.array(img)
        mask = np.any(arr != 255, axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))

    def draw_single(img: Image.Image, boxes, masks, indices, out_path: Path):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        color = "red"

        def draw_mask_contour(mask_path):
            if mask_path is None:
                return False
            try:
                m = np.array(Image.open(mask_path).convert("L"))
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                drawn = False
                for c in contours:
                    c = c.squeeze()
                    if c.ndim != 2 or c.shape[0] < 2:
                        continue
                    ax.plot(c[:, 0], c[:, 1], color=color, linewidth=1.0)
                    drawn = True
                return drawn
            except Exception:
                return False

        for idx_box in indices:
            b = boxes[idx_box]["box"]
            drawn = draw_mask_contour(masks[idx_box])
            if not drawn:
                rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1.0, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            ax.text(
                cx,
                cy,
                f"{idx_box}",
                color=color,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=color, boxstyle="round,pad=0.1"),
            )
        ax.axis("off")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img_out = Image.fromarray(rgba[..., :3])
        img_out = trim_whitespace(img_out)
        img_out.save(out_path)
        plt.close(fig)

    # 結合してマッチング線ありの最終可視化
    canvas = Image.new("RGB", (w0 + w1, max(h0, h1)), (255, 255, 255))
    canvas.paste(img0, (0, 0))
    canvas.paste(img1, (w0, 0))
    canvas_w, canvas_h = canvas.size
    dpi = 100
    fig, ax = plt.subplots(figsize=(canvas_w / dpi, canvas_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.imshow(canvas)
    lines_to_draw = []
    def draw_mask_contour(
        ax_local,
        mask_path,
        x_offset=0,
        scale=1.0,
        line_color="red",
        fill_alpha=0.35,
        bbox=None,
        full_size=None,
        id_arr=None,
        idx=None,
    ):
        if mask_path is None and id_arr is None:
            return False
        try:
            if id_arr is not None and idx is not None:
                if id_arr.ndim != 2:
                    return False
                m = (id_arr == idx).astype(np.uint8) * 255
            else:
                m = np.array(Image.open(mask_path).convert("L"))
            use_full = False
            if full_size is not None:
                fh, fw = full_size
                use_full = (m.shape[0] == fh and m.shape[1] == fw)
            if use_full:
                if scale != 1.0:
                    h, w = m.shape[:2]
                    m = cv2.resize(
                        m,
                        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_NEAREST,
                    )
                offset_x = x_offset
                offset_y = 0
                target_w = m.shape[1]
                target_h = m.shape[0]
            else:
                if bbox is None:
                    return False
                x0, y0, x1, y1 = bbox
                target_w = max(1, int(round(x1 - x0)))
                target_h = max(1, int(round(y1 - y0)))
                if m.shape[0] != target_h or m.shape[1] != target_w:
                    m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                offset_x = x_offset + x0
                offset_y = y0
            if fill_alpha > 0.0:
                rgb = mcolors.to_rgb(line_color)
                overlay = np.zeros((target_h, target_w, 4), dtype=np.float32)
                overlay[..., 0] = rgb[0]
                overlay[..., 1] = rgb[1]
                overlay[..., 2] = rgb[2]
                overlay[..., 3] = (m > 0).astype(np.float32) * fill_alpha
                ax_local.imshow(
                    overlay,
                    extent=(offset_x, offset_x + target_w, offset_y + target_h, offset_y),
                    interpolation="nearest",
                    zorder=3,
                )
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            drawn = False
            for c in contours:
                c = c.squeeze()
                if c.ndim != 2 or c.shape[0] < 2:
                    continue
                ax_local.plot(c[:, 0] + offset_x, c[:, 1] + offset_y, color=line_color, linewidth=1.0, zorder=5)
                drawn = True
            return drawn
        except Exception:
            return False

    pastel_palette = [
        "#ff8fb1", "#8ec5ff", "#98e27e", "#ffd166", "#c8a2ff",
        "#66d9cc", "#ffb26b", "#7fc8a9", "#f39ac7", "#7adfa5",
        "#d28fe8", "#9bd16f", "#84c5d8", "#f7a8a8", "#b7db6f",
    ]

    for i0, i1 in pairs:
        if i0 >= len(boxes0) or i1 >= len(boxes1):
            continue
        if vis_style == "pair_color":
            pair_color = pastel_palette[len(lines_to_draw) % len(pastel_palette)]
            fill_alpha = 0.35
        else:
            pair_color = "red"
            fill_alpha = 0.0
        b0 = [v * scale0 for v in boxes0[i0]["box"]]
        b1 = [v * scale1 for v in boxes1[i1]["box"]]
        cx0 = (b0[0] + b0[2]) / 2
        cy0 = (b0[1] + b0[3]) / 2
        cx1 = (b1[0] + b1[2]) / 2
        cy1 = (b1[1] + b1[3]) / 2
        drawn0 = draw_mask_contour(
            ax,
            masks0[i0],
            x_offset=0,
            scale=scale0,
            line_color=pair_color,
            fill_alpha=fill_alpha,
            bbox=b0,
            full_size=(orig_h0, orig_w0),
            id_arr=id_arr0,
            idx=i0,
        )
        drawn1 = draw_mask_contour(
            ax,
            masks1[i1],
            x_offset=w0,
            scale=scale1,
            line_color=pair_color,
            fill_alpha=fill_alpha,
            bbox=b1,
            full_size=(orig_h1, orig_w1),
            id_arr=id_arr1,
            idx=i1,
        )
        if not drawn0:
            rect0 = patches.Rectangle(
                (b0[0], b0[1]),
                b0[2] - b0[0],
                b0[3] - b0[1],
                linewidth=1.0,
                edgecolor=pair_color,
                facecolor=mcolors.to_rgba(pair_color, fill_alpha),
                zorder=5,
            )
            ax.add_patch(rect0)
        if not drawn1:
            rect1 = patches.Rectangle(
                (w0 + b1[0], b1[1]),
                b1[2] - b1[0],
                b1[3] - b1[1],
                linewidth=1.0,
                edgecolor=pair_color,
                facecolor=mcolors.to_rgba(pair_color, fill_alpha),
                zorder=5,
            )
            ax.add_patch(rect1)
        if show_index:
            ax.text(
                cx0,
                cy0,
                f"{i0}",
                color=pair_color,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=pair_color, boxstyle="round,pad=0.1"),
                zorder=1000,
            )
            ax.text(
                w0 + cx1,
                cy1,
                f"{i1}",
                color=pair_color,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=pair_color, boxstyle="round,pad=0.1"),
                zorder=1000,
            )
        if answer_pairs is None:
            line_color = "lime"
        else:
            line_color = "lime" if (i0, i1) in answer_pairs else "red"
        lines_to_draw.append((cx0, cy0, w0 + cx1, cy1, line_color))

    for (x0, y0, x1, y1, line_color) in lines_to_draw:
        ax.plot([x0, x1], [y0, y1], color=line_color, linewidth=2.0, zorder=10)
    ax.axis("off")
    fig.canvas.draw()
    rgba2 = np.asarray(fig.canvas.buffer_rgba())
    full_img2 = Image.fromarray(rgba2[..., :3])
    vis_path = match_dir / "best_match_vis.png"
    full_img2.save(vis_path)
    plt.close(fig)
    print(f"visualization saved: {vis_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", required=True, help="path to image0")
    parser.add_argument("--image1", required=True, help="path to image1")
    parser.add_argument("--manual", required=False, help="path to manual.txt")
    parser.add_argument("--result_root", default="/home/yu/Otani/result", help="result root dir")
    parser.add_argument("--prompts0", nargs="+", default=["wooden board"], help="prompts for image0")
    parser.add_argument("--prompts1", nargs="+", default=["wooden board"], help="prompts for image1")
    parser.add_argument("--alpha", type=float, default=2.0, help="weight for appearance energy")
    parser.add_argument("--beta", type=float, default=1.0, help="weight for geometry energy")
    parser.add_argument("--confidence", type=float, default=0.5, help="SAM3 confidence threshold")
    parser.add_argument(
        "--vis_style",
        choices=["pair_color", "red_contour"],
        default="pair_color",
        help="visualization style for boxes/masks",
    )
    parser.add_argument("--show_index", action="store_true", help="show part indices on visualization")
    parser.add_argument(
        "--conda_sh",
        default="~/anaconda3/etc/profile.d/conda.sh",
        help="conda.sh のパス (source 用)。見つからない場合に anaconda3/miniconda3 を自動で探します。",
    )
    parser.add_argument(
        "--perspective_cmd",
        default="cd /home/yu/Otani/PerspectiveFields && python demo.py --image {image} --save-extrinsic {extrinsic}",
        help="PerspectiveFields 実行コマンドのテンプレート (既定は demo.py)。{image},{output},{extrinsic} が利用可能。",
    )
    parser.add_argument(
        "--use_id_mask",
        action="store_true",
        help="Use precomputed IdMask masks if available (skip SAM3).",
    )
    parser.add_argument(
        "--metric3d_cmd",
        default="cd /home/yu/Otani/Metric3D && python demo.py --image {image} --model metric3d_vit_small --save-npy {depth} --save-png {output}/depth.png --save-cmap {output}/depth_color.png",
        help="Metric3Dv2 実行コマンドのテンプレート (既定は demo.py)。{image},{output},{depth},{intrinsic} が利用可能。",
    )
    parser.add_argument(
        "--wildcamera_cmd",
        default="cd /home/yu/Otani/WildCamera && python demo/demo_inference.py {image}",
        help="WildCamera 実行コマンドのテンプレート (焦点距離推定)。{image} が利用可能。",
    )
    parser.add_argument(
        "--use_wildcamera",
        action="store_true",
        default=True,
        help="WildCamera で焦点距離を推定して Metric3D 深度をスケールする（デフォルト有効）。--use_wildcamera を付けなくても有効、無効化したい場合は --no_wildcamera を用意していません。",
    )
    parser.add_argument(
        "--contact_epsilon",
        type=float,
        default=0.01,
        help="AABB 接触判定の許容距離（メートル換算スケール）。",
    )
    args = parser.parse_args()

    # prompts
    prompts0 = args.prompts0
    prompts1 = args.prompts1

    # models
    model, dino_model, dino_preprocess, dino_device = load_models()

    # result dir
    run_dir = Path(args.result_root) / f"{Path(args.image0).stem}_{Path(args.image1).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存先: {run_dir}")

    # prompts & embeddings
    base0, cnt0 = run_prompts(
        args.image0,
        prompts0,
        run_dir,
        model,
        dino_model,
        dino_preprocess,
        dino_device,
        confidence=args.confidence,
        annot_name="vis_image0_annot.png",
        draw_index=True,
        use_id_mask=args.use_id_mask,
    )
    base1, cnt1 = run_prompts(
        args.image1,
        prompts1,
        run_dir,
        model,
        dino_model,
        dino_preprocess,
        dino_device,
        confidence=args.confidence,
        annot_name="vis_image1_annot.png",
        draw_index=True,
        use_id_mask=args.use_id_mask,
    )
    if cnt0 == 0:
        raise RuntimeError("image0 で検出されたボックスがありません。prompts0 や --confidence を見直してください。")
    if cnt1 == 0:
        raise RuntimeError("image1 で検出されたボックスがありません。prompts1 や --confidence を見直してください。")

    # VGGT による 3D 解析 (image1)
    if args.beta != 0.0:
        try:
            point_map, preprocess_info = run_vggt_pointmap(args.image1, run_dir / "VGGT")
            rel_path = compute_relations_from_pointmap(base1, point_map, preprocess_info)
        except Exception as e:
            print(f"3D パイプライン(VGGT)でエラーが発生しました: {e}")
            rel_path = None
    else:
        rel_path = None

    # load feats
    vecs0, labels0 = load_all_feat_vectors(base0, allow_prompts=prompts0)
    vecs1, labels1 = load_all_feat_vectors(base1, allow_prompts=prompts1)
    if vecs0.numel() == 0 or vecs1.numel() == 0:
        raise RuntimeError("特徴が空です。")

    # energy search with timing
    t0 = time.time()
    result = energy_search(
        vecs0,
        vecs1,
        labels0,
        labels1,
        base1,
        args.manual,
        run_dir,
        contact_eps=args.contact_epsilon,
        alpha=args.alpha,
        beta=args.beta,
    )
    if result is None:
        print("energy_search failed or skipped; matching result not available.")
        return
    best_match, match_dir = result
    elapsed = time.time() - t0
    print(f"energy_search elapsed: {elapsed:.5f}s")

    # visualize
    visualize_match(
        best_match,
        base0,
        base1,
        args.image0,
        args.image1,
        match_dir,
        vis_style=args.vis_style,
        show_index=args.show_index,
    )


if __name__ == "__main__":
    # tfloat32/bfloat16 設定
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    main()