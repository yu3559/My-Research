#!/usr/bin/env python3
import argparse
import ast
import base64
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path


def _collect_mask_id_to_name(node, acc: dict) -> None:
    """hierarchyList の各ノードに付く maskId（パート番号）→ name を収集。"""
    if isinstance(node, dict):
        mid = node.get("maskId")
        if mid is not None:
            try:
                acc[int(mid)] = str(node.get("name", ""))
            except (TypeError, ValueError):
                pass
        for ch in node.get("children") or []:
            _collect_mask_id_to_name(ch, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_mask_id_to_name(item, acc)


def hierarchy_instance_and_part_names(row, n_parts: int):
    """
    PartNeXt の hierarchyList からインスタンス名と、maskId 順の部品名リストを返す。
    maskId が無いパートは「パーツi」で埋める。
    """
    raw = row.get("hierarchyList")
    if not raw:
        return "object", [f"パーツ{i}" for i in range(n_parts)]
    try:
        h = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return "object", [f"パーツ{i}" for i in range(n_parts)]
    root = h[0] if isinstance(h, list) else h
    instance = str(root.get("name", "object"))
    acc = {}
    _collect_mask_id_to_name(root, acc)
    part_names = []
    for i in range(n_parts):
        part_names.append(acc.get(i) if acc.get(i) else f"パーツ{i}")
    return instance, part_names


def _png_file_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_openai_assembly_manual(
    *,
    api_key: str,
    model: str,
    assembled_annot: Path,
    disassembled_annot: Path,
    instance_name: str,
    part_names: list[str],
    out_path: Path,
) -> None:
    """
    組立説明テキストを OpenAI Chat Completions（ビジョン）で生成し out_path に保存する。
    依存: pip install openai
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("OpenAI API を使うには: pip install openai") from e

    part_lines = "\n".join(
        f"パーツ{i}：{part_names[i]}" for i in range(len(part_names))
    )
    text = (
        f"添付の1枚目の画像は、{instance_name}（インスタンス名）の画像です。"
        f"2枚目の画像は、1枚目の{instance_name}（インスタンス名）をそれぞれの部品に分けた状態の画像です。"
        "2枚の画像で、同じ部品は同じインデックスを表示しています。"
        "この物体の組み立て手順を、説明書風に書いてください。\n"
        "出力では、太文字にするためのアスタリスクを使わないでください。"
        "パーツの名称は、一般的な名称ではなく「パーツ0」「パーツ1」のように書いてください。\n"
        "また、それぞれのパーツは次のとおりです。\n"
        f"{part_lines}\n\n"
        "以下が出力形式\n"
        "------------------------------\n"
        "STEP1：\n"
        "1.〇〇\n"
        "2.〇〇\n"
        "...\n"
        "STEP2：\n"
        "1. ....\n"
        "....\n"
        "STEPN：\n"
    )

    url1 = _png_file_to_data_url(assembled_annot)
    url2 = _png_file_to_data_url(disassembled_annot)

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ],
        max_tokens=4096,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI の応答が空です")
    out_path.write_text(content, encoding="utf-8")


def load_annotations(ann_dir: Path):
    try:
        import pyarrow.ipc as ipc
    except Exception as e:
        raise RuntimeError("pyarrow is required to read PartNeXt annotations") from e

    rows = []
    for arrow_path in sorted(ann_dir.glob("data-*.arrow")):
        with arrow_path.open("rb") as f:
            reader = ipc.open_stream(f)
            table = reader.read_all()
            for record in table.to_pylist():
                rows.append(record)
    return rows


def _scene_scale_hint(scene) -> float:
    try:
        import numpy as np

        b = scene.bounds
        if b is None or getattr(b, "shape", None) != (2, 3):
            return 1.0
        d = b[1] - b[0]
        L = float(np.linalg.norm(d))
        return L if L > 1e-12 else 1.0
    except Exception:
        return 1.0


def _path_like_to_trimesh(path, scale_hint: float):
    """
    Path / Path3D / Path2D をレンダ可能な三角メッシュに近似的に変換する。
    1) 塗りつぶしポリゴンがあれば extrude
    2) discrete 折れ線を細い円柱でつなぐ
    3) 共面なら to_2D して extrude
    """
    try:
        import numpy as np
        import trimesh
        from trimesh.path.path import Path as TrimeshPath
    except Exception:
        return None

    if not isinstance(path, TrimeshPath):
        return None

    radius = max(float(scale_hint) * 1e-4, 1e-7)
    height = max(radius * 4.0, float(scale_hint) * 1e-5)

    polys = getattr(path, "polygons_full", None)
    if polys:
        try:
            extruded = path.extrude(height=height)
            if extruded is None:
                pass
            elif isinstance(extruded, list):
                if extruded:
                    m = trimesh.util.concatenate(extruded)
                    if len(m.faces) > 0:
                        return m
            elif len(extruded.faces) > 0:
                return extruded
        except Exception:
            pass

    meshes = []
    try:
        for pts in path.discrete:
            pts = np.asanyarray(pts, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
            elif pts.shape[1] != 3:
                continue
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                seg = b - a
                h = float(np.linalg.norm(seg))
                if h < 1e-12:
                    continue
                cyl = trimesh.creation.cylinder(radius=radius, height=h, sections=8)
                direction = seg / h
                z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                T = trimesh.geometry.align_vectors(z, direction).copy()
                T[:3, 3] = (a + b) * 0.5
                cyl.apply_transform(T)
                meshes.append(cyl)
    except Exception:
        meshes = []

    if meshes:
        return trimesh.util.concatenate(meshes)

    try:
        p2d, to_3d = path.to_2D(check=False)
        polys2 = getattr(p2d, "polygons_full", None)
        if polys2:
            extruded = p2d.extrude(height=height)
            if extruded is None:
                return None
            if isinstance(extruded, list):
                if not extruded:
                    return None
                extruded = trimesh.util.concatenate(extruded)
            extruded.apply_transform(to_3d)
            if len(extruded.faces) > 0:
                return extruded
    except Exception:
        pass
    return None


def _convert_geometry_to_trimesh(geom, scale_hint: float):
    try:
        import numpy as np
        import trimesh
        from trimesh.path.path import Path as TrimeshPath
        from trimesh.points import PointCloud
    except Exception:
        return None

    faces = getattr(geom, "faces", None)
    if faces is not None and len(faces) > 0:
        verts = getattr(geom, "vertices", None)
        if verts is None or len(verts) == 0:
            return None
        if isinstance(geom, trimesh.Trimesh):
            return geom
        try:
            return trimesh.Trimesh(
                vertices=np.asarray(geom.vertices, dtype=np.float64),
                faces=np.asarray(geom.faces, dtype=np.int64),
            )
        except Exception:
            return None

    if isinstance(geom, PointCloud):
        pts = geom.vertices
        if pts is None or len(pts) == 0:
            return None
        if len(pts) >= 4:
            try:
                hull = geom.convex_hull
                if hull is not None and len(hull.faces) > 0:
                    return hull
            except Exception:
                pass
        r = max(float(scale_hint) * 1e-4, 1e-7)
        try:
            sph = trimesh.creation.icosphere(subdivisions=1, radius=r)
            sph.apply_translation(geom.centroid)
            return sph
        except Exception:
            return None

    if isinstance(geom, TrimeshPath):
        return _path_like_to_trimesh(geom, scale_hint)

    return None


def load_partnext_mesh(glb_path: Path):
    try:
        import partnext.io as pio
        import trimesh
    except Exception as e:
        raise RuntimeError("partnext package is required to load GLB meshes") from e
    scene = pio.load_glb(str(glb_path))
    scale = _scene_scale_hint(scene)
    # scene.geometry の列順と masks の mesh インデックスを一致させるため、
    # 各スロットに必ず 1 エントリを割り当てる（変換失敗時は空メッシュ）
    meshes = []
    if hasattr(scene, "geometry"):
        for geom in scene.geometry.values():
            tm = _convert_geometry_to_trimesh(geom, scale)
            if tm is None or len(getattr(tm, "faces", [])) == 0:
                meshes.append(trimesh.Trimesh())
            else:
                meshes.append(tm)
    return meshes


def build_parts(meshes, masks):
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError("trimesh is required to build part meshes") from e
    parts = []
    for part_id in sorted(masks.keys(), key=lambda x: int(x)):
        part_meshes = []
        for mesh_idx in masks[part_id]:
            face_indices = masks[part_id][mesh_idx]
            try:
                mesh_i = int(mesh_idx)
            except Exception:
                continue
            if mesh_i < 0 or mesh_i >= len(meshes):
                continue
            mesh = meshes[mesh_i]
            # guard against out-of-range face indices
            face_indices = [i for i in face_indices if 0 <= i < len(mesh.faces)]
            if not face_indices:
                continue
            part_meshes.append(mesh.submesh([face_indices], append=True))
        if part_meshes:
            part_mesh = trimesh.util.concatenate(part_meshes)
            parts.append(part_mesh)
    return parts


def write_answer(out_dir: Path):
    id_map_path = out_dir / "id_mask_map.json"
    if not id_map_path.exists():
        return
    try:
        id_map = json.loads(id_map_path.read_text())
        part_ids = sorted(id_map.values())
        n_parts = len(part_ids)
    except Exception:
        n_parts = 0
    if n_parts <= 0:
        return
    answer_path = out_dir / "answer.txt"
    with answer_path.open("w", encoding="utf-8") as f:
        for i in range(n_parts):
            f.write(f"({i},{i})\n")


def write_id_mask_annot(out_dir: Path):
    return


def draw_part_indices(image_path: Path, id_npy_path: Path, out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except Exception:
        return
    if not image_path.exists() or not id_npy_path.exists():
        return
    try:
        id_arr = np.load(id_npy_path)
    except Exception:
        return
    if id_arr.ndim != 2:
        return
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    max_id = int(id_arr.max()) if id_arr.size else -1
    for part_id in range(max_id + 1):
        ys, xs = np.where(id_arr == part_id)
        if ys.size == 0 or xs.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        text = str(part_id)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 2
        x0 = max(0, cx - tw // 2 - pad)
        y0 = max(0, cy - th // 2 - pad)
        x1 = min(img.width, x0 + tw + pad * 2)
        y1 = min(img.height, y0 + th + pad * 2)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(255, 0, 0), width=1)
        draw.text((x0 + pad, y0 + pad), text, fill=(255, 0, 0), font=font)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default="/home/yu/Otani/PartNeXt_lib")
    parser.add_argument("--glb_dir", default="/home/yu/Otani/PartNeXt_lib/glbs")
    parser.add_argument("--out_root", default="/home/yu/Otani/assets/partnext_renders")
    parser.add_argument("--blender_bin", default="blender")
    parser.add_argument("--blender_script", default="/home/yu/Otani/blender_render_data_v0.py")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--max_parts", type=int, default=0, help="skip models with parts count > this value")
    parser.add_argument("--only_glb_id", default="")
    parser.add_argument("--keep_objs", action="store_true")
    parser.add_argument(
        "--save_stage_blends",
        action="store_true",
        help="blender_render_data_v0.py に渡し、組立前・組立後のレンダ直前シーンを .blend で保存する",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="レンダ後に OpenAI API で組立説明テキストを生成し assembly_manual_openai.txt に保存する（要 pip install openai と API キー）",
    )
    parser.add_argument("--openai_model", default="gpt-4o-mini", help="ビジョン対応モデル名")
    parser.add_argument(
        "--openai_key_env",
        default="OPENAI_API_KEY",
        help="API キーを読む環境変数名（既定: OPENAI_API_KEY）",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    glb_dir = Path(args.glb_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # allow importing local PartNeXt_lib/partnext without pip install
    if str(ann_dir) not in sys.path:
        sys.path.insert(0, str(ann_dir))

    rows = load_annotations(ann_dir)
    if args.max_objects > 0:
        rows = rows[: args.max_objects]

    for row in rows:
        glb_id = row.get("glb_id")
        model_id = row.get("model_id")
        type_id = row.get("type_id")
        if not model_id or not type_id:
            continue
        if args.only_glb_id and args.only_glb_id not in (glb_id, model_id, f"{type_id}_{model_id}"):
            continue
        out_dir = out_root / f"{type_id}_{model_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        glb_path = glb_dir / type_id / f"{model_id}.glb"
        if not glb_path.exists():
            continue

        masks = row.get("masks")
        if isinstance(masks, str):
            masks = ast.literal_eval(masks)
        if not masks:
            continue

        meshes = load_partnext_mesh(glb_path)
        parts = build_parts(meshes, masks)
        if not parts:
            if out_dir.exists() and not any(out_dir.iterdir()):
                out_dir.rmdir()
            continue
        if args.max_parts > 0 and len(parts) > args.max_parts:
            if out_dir.exists() and not any(out_dir.iterdir()):
                out_dir.rmdir()
            continue

        obj_dir = out_dir / "objs"
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
        obj_dir.mkdir(parents=True, exist_ok=True)
        for i, part_mesh in enumerate(parts):
            part_mesh.export(obj_dir / f"new-{i}.obj")

        cmd = [
            args.blender_bin,
            "--background",
            "--python",
            args.blender_script,
            "--",
            "--obj_dir",
            str(obj_dir),
            "--out_dir",
            str(out_dir),
            "--image_size",
            str(args.image_size),
        ]
        if args.save_stage_blends:
            cmd.append("--save_stage_blends")
        subprocess.run(cmd, check=True)
        write_answer(out_dir)
        write_id_mask_annot(out_dir)
        draw_part_indices(
            out_dir / "assembled.png",
            out_dir / "id_mask_assembled.npy",
            out_dir / "assembled_annot.png",
        )
        draw_part_indices(
            out_dir / "disassembled.png",
            out_dir / "id_mask_disassembled.npy",
            out_dir / "disassembled_annot.png",
        )

        instance_name, part_names = hierarchy_instance_and_part_names(row, len(parts))
        if args.openai:
            api_key = os.environ.get(args.openai_key_env, "").strip()
            if not api_key:
                print(
                    f"[warn] --openai ですが環境変数 {args.openai_key_env} が空のためスキップ",
                    file=sys.stderr,
                )
            else:
                try:
                    call_openai_assembly_manual(
                        api_key=api_key,
                        model=args.openai_model,
                        assembled_annot=out_dir / "assembled_annot.png",
                        disassembled_annot=out_dir / "disassembled_annot.png",
                        instance_name=instance_name,
                        part_names=part_names,
                        out_path=out_dir / "assembly_manual_openai.txt",
                    )
                except Exception as e:
                    print(f"[warn] OpenAI 呼び出し失敗: {e}", file=sys.stderr)

        if not args.keep_objs:
            shutil.rmtree(obj_dir)


if __name__ == "__main__":
    main()
