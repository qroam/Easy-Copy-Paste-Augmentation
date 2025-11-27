"""
Copy-Paste Augmentation Studio — Streamlit Frontend

Requirements (pip):
    streamlit
    pyyaml
    pillow

Optional (better YAML editing but not required):
    ruamel.yaml

Run:
    streamlit run app.py

Notes:
- This app does NOT implement the augmentation logic; it calls your provided Python
  function: main(work_dir, output_folder_name, config_filename).
- Set the "Augmentor entry point" to a fully-qualified import path where `main` lives,
  e.g. `my_pkg.copypaste.entry:main` or `my_pkg.copypaste.entry` (we'll resolve `main`).
- Progress is estimated by counting new files in the output dataset folder while the
  job runs. If your backend exposes richer progress, you can wire it in easily.
"""

from __future__ import annotations
import os
import io
import re
import sys
import time
import json
import glob
import uuid
import shutil
import types
import queue
import base64
import threading
import importlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import yaml
from PIL import Image, ImageDraw

# =============================
# --------- SETTINGS ----------
# =============================
# You can change these defaults in the UI Settings panel
DEFAULT_RULES_DIR = str(Path.cwd() / "rules")
DEFAULT_MATERIALS_DIR = str(Path.cwd() / "materials")
DEFAULT_OUTPUTS_DIR = str(Path.cwd() / "datasets")
DEFAULT_HISTORY_PATH = str(Path.cwd() / "history.jsonl")
DEFAULT_AUGMENTOR_ENTRY = "your_package.copypaste.entry:main"  # edit in UI

# =============================
# ---------- MODELS -----------
# =============================
@dataclass
class MaterialSet:
    class_name: str
    folder: str
    created_at: str
    updated_at: str
    sample_count: int

@dataclass
class BuildRecord:
    run_id: str
    dataset_name: str
    background_dir: str
    material_sets: List[str]  # list of class names or folders
    rule_file: str
    created_at: str
    params: Dict[str, Any]
    output_dir: str

# =============================
# ------- PERSISTENCE ---------
# =============================
class HistoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def append(self, rec: BuildRecord):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def read_all(self) -> List[BuildRecord]:
        if not self.path.exists():
            return []
        recs = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    recs.append(BuildRecord(**d))
                except Exception:
                    continue
        # Sort newest first
        recs.sort(key=lambda r: r.created_at, reverse=True)
        return recs

# =============================
# --------- UTILITIES ---------
# =============================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files: List[Path] = []
    for e in exts:
        files.extend(Path(folder).glob(e))
    return sorted(files)


def load_yaml_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="replace")


def validate_yaml(content: str) -> Tuple[bool, Optional[str]]:
    try:
        _ = yaml.safe_load(content) if content.strip() else {}
        return True, None
    except Exception as e:
        return False, str(e)


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def count_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.rglob("*.*"))


def resolve_main(entry: str):
    """Resolve an entry string to a callable.
    Accepted formats:
        - "pkg.module.submod:main"
        - "pkg.module.submod" (must expose attr `main`)
    Returns the `main` callable.
    """
    fn_name = None
    mod_path = entry
    if ":" in entry:
        mod_path, fn_name = entry.split(":", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name or "main")
    if not callable(fn):
        raise RuntimeError(f"Resolved object is not callable: {entry}")
    return fn


# =============================
# ------- STATE & CACHE -------
# =============================
if "settings" not in st.session_state:
    st.session_state.settings = {
        "rules_dir": DEFAULT_RULES_DIR,
        "materials_dir": DEFAULT_MATERIALS_DIR,
        "outputs_dir": DEFAULT_OUTPUTS_DIR,
        "history_path": DEFAULT_HISTORY_PATH,
        "augmentor_entry": DEFAULT_AUGMENTOR_ENTRY,
    }

if "busy" not in st.session_state:
    st.session_state.busy = False

if "progress" not in st.session_state:
    st.session_state.progress = 0.0

if "running_thread" not in st.session_state:
    st.session_state.running_thread = None

# =============================
# ------------ UI -------------
# =============================
st.set_page_config(page_title="Copy-Paste Augmentation Studio", layout="wide")
st.title("🧩 Copy-Paste Augmentation Studio")

with st.expander("⚙️ Settings", expanded=False):
    cols = st.columns(2)
    with cols[0]:
        st.session_state.settings["rules_dir"] = st.text_input(
            "Rules directory", st.session_state.settings["rules_dir"]
        )
        st.session_state.settings["materials_dir"] = st.text_input(
            "Materials root directory", st.session_state.settings["materials_dir"]
        )
        st.session_state.settings["outputs_dir"] = st.text_input(
            "Datasets output root", st.session_state.settings["outputs_dir"]
        )
    with cols[1]:
        st.session_state.settings["history_path"] = st.text_input(
            "History JSONL path", st.session_state.settings["history_path"]
        )
        st.session_state.settings["augmentor_entry"] = st.text_input(
            "Augmentor entry point (module[:func])",
            st.session_state.settings["augmentor_entry"],
            help="Import path resolving to a callable main(work_dir, output_folder_name, config_filename)",
        )

# Init persistence
RULES_DIR = ensure_dir(st.session_state.settings["rules_dir"])  # type: ignore
MATERIALS_DIR = ensure_dir(st.session_state.settings["materials_dir"])  # type: ignore
OUTPUTS_DIR = ensure_dir(st.session_state.settings["outputs_dir"])  # type: ignore
HISTORY = HistoryStore(st.session_state.settings["history_path"])  # type: ignore

# =============== TAB LAYOUT ===============

rules_tab, mats_tab, build_tab, gallery_tab = st.tabs([
    "📜 贴图规则管理",
    "🗂️ 贴图素材集管理",
    "🏭 样本生产（工作流）",
    "🖼️ 画廊",
])

# -----------------------------------------
# 1) 贴图规则管理
# -----------------------------------------
with rules_tab:
    st.subheader("📜 贴图规则管理")
    left, right = st.columns([1, 2])

    # List existing YAML files
    with left:
        st.caption("规则文件（.yaml / .yml）")
        yaml_files = sorted(list(RULES_DIR.glob("*.y*ml")))
        names = [f.name for f in yaml_files]
        selected_name = st.selectbox("选择规则文件", options=["<新建>"] + names)

        if selected_name == "<新建>":
            new_name = st.text_input("新建规则文件名（含扩展名 .yaml）", "rule_example.yaml")
            if st.button("创建空白规则文件"):
                target = RULES_DIR / new_name
                if target.exists():
                    st.warning("已有同名文件。")
                else:
                    template = (
                        "# 示例模板\n"
                        "plasticbag:\n  - D:/贴图样本集/cropped_objects/class_0\n\n"
                        "number:\n  plasticbag: [1, 3]\n\n"
                        "rules:\n  plasticbag:\n    size:\n      reference: _\n      rule: mean\n      scale: [0.03, 0.04]\n    position:\n      reference: _\n      value: [0.40, 0.05, 0.55, 0.95]\n"
                    )
                    save_text(target, template)
                    st.success(f"已创建 {target}")
                    st.experimental_rerun()
        else:
            st.info(f"当前文件：{selected_name}")
            # Rename / delete helpers
            new_basename = st.text_input("重命名为", selected_name)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("重命名") and new_basename != selected_name:
                    src = RULES_DIR / selected_name
                    dst = RULES_DIR / new_basename
                    if dst.exists():
                        st.error("目标文件已存在。")
                    else:
                        src.rename(dst)
                        st.success("已重命名。")
                        st.experimental_rerun()
            with c2:
                if st.button("删除此规则文件", type="secondary"):
                    (RULES_DIR / selected_name).unlink(missing_ok=True)
                    st.success("已删除。")
                    st.experimental_rerun()

    # Editor
    with right:
        if selected_name == "<新建>":
            st.caption("（新建模式）在左侧创建文件后再进行编辑。")
        else:
            target = RULES_DIR / selected_name
            content = load_yaml_text(target)
            text = st.text_area("编辑 YAML 规则", value=content, height=420)
            valid, err = validate_yaml(text)
            if valid:
                st.success("YAML 语法校验通过。")
            else:
                st.error(f"YAML 语法错误：{err}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存修改", disabled=not valid):
                    save_text(target, text)
                    st.toast("已保存。", icon="✅")
            with c2:
                st.download_button("下载此规则文件", data=text, file_name=selected_name, mime="text/yaml")

# -----------------------------------------
# 2) 贴图素材集管理
# -----------------------------------------
with mats_tab:
    st.subheader("🗂️ 贴图素材集管理")

    # Register an existing folder as a material set
    with st.expander("📁 绑定现有文件夹为素材集", expanded=True):
        class_name = st.text_input("素材集类名 (e.g., plasticbag)")
        folder_path = st.text_input("素材集文件夹路径", value=str(MATERIALS_DIR / "plasticbag"))
        if st.button("绑定为素材集"):
            if not class_name:
                st.error("请输入类名。")
            else:
                folder = ensure_dir(folder_path)
                # Count samples
                sc = len(list_images(folder))
                meta = MaterialSet(
                    class_name=class_name,
                    folder=str(folder),
                    created_at=now_str(),
                    updated_at=now_str(),
                    sample_count=sc,
                )
                # Write a metadata file under the folder
                (folder / "_material_meta.json").write_text(
                    json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                st.success(f"已绑定：{class_name} -> {folder}")

    # Uploader to add images into a class folder
    with st.expander("📤 向素材集中新增图片", expanded=False):
        # discover known sets
        known_sets = []
        for p in MATERIALS_DIR.rglob("_material_meta.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                known_sets.append((d["class_name"], Path(d["folder"])) )
            except Exception:
                pass
        if not known_sets:
            st.info("尚未绑定任何素材集。请先在上面绑定。")
        else:
            choices = [f"{cn}  —  {fp}" for cn, fp in known_sets]
            sel = st.selectbox("选择素材集", options=choices)
            files = st.file_uploader("选择图片 (可多选)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], accept_multiple_files=True)
            if st.button("上传到素材集") and files:
                idx = choices.index(sel)
                cn, fp = known_sets[idx]
                saved = 0
                for f in files:
                    suffix = Path(f.name).suffix or ".jpg"
                    out = fp / f"{uuid.uuid4().hex}{suffix}"
                    out.write_bytes(f.read())
                    saved += 1
                # update meta
                meta_path = fp / "_material_meta.json"
                d = json.loads(meta_path.read_text(encoding="utf-8"))
                d["updated_at"] = now_str()
                d["sample_count"] = len(list_images(fp))
                meta_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success(f"已保存 {saved} 张图片到 {cn}")

    # Overview table
    with st.expander("📊 素材集一览", expanded=True):
        rows = []
        for meta in MATERIALS_DIR.rglob("_material_meta.json"):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                rows.append(d)
            except Exception:
                continue
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("暂无素材集。")

# -----------------------------------------
# 3) 样本生产（主要工作流）
# -----------------------------------------
with build_tab:
    st.subheader("🏭 样本生产（主要工作流）")

    c1, c2 = st.columns(2)
    with c1:
        dataset_name = st.text_input("构造的数据集名称", value=f"ds_{datetime.now().strftime('%m%d_%H%M')}")
        background_dir = st.text_input("背景图所在目录 (work_dir)", value=str(Path.cwd() / "backgrounds"))
        rule_file = st.text_input("贴图规则文件 (.yaml)", value=str((RULES_DIR / "rule_example.yaml")))
        per_bg_count = st.number_input("每张背景图生成的样本数量", min_value=1, max_value=50, value=3)
        extra_params = st.text_area("其他参数 (JSON，可留空)", value="{}")
    with c2:
        st.caption("选择/确认素材集（作为说明存档，后端按规则文件查找）")
        # show known sets for user's reference
        known_sets = []
        for meta in MATERIALS_DIR.rglob("_material_meta.json"):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                known_sets.append(d)
            except Exception:
                pass
        mat_labels = [f"{d['class_name']} ({d['sample_count']})" for d in known_sets]
        selected_idx = st.multiselect("素材集（多选仅用于记录）", options=list(range(len(known_sets))), format_func=lambda i: mat_labels[i])
        selected_sets = [known_sets[i]["class_name"] for i in selected_idx]
        output_dir = ensure_dir(OUTPUTS_DIR / dataset_name)
        st.info(f"输出目录：{output_dir}")

    # Run controller
    run_button = st.button("🚀 合成 (调用后端)", disabled=st.session_state.busy)

    def run_job():
        st.session_state.busy = True
        try:
            # Resolve entry
            entry = st.session_state.settings["augmentor_entry"]
            fn = resolve_main(entry)

            # Prepare params
            params = json.loads(extra_params or "{}")
            params.update({
                "work_dir": background_dir,
                "output_folder_name": str(output_dir),
                "config_filename": rule_file,
                # If your backend supports per_bg_count as an arg, you can include it here too
                # e.g., "per_bg_count": per_bg_count
            })

            # Estimate total expected files using background count * per_bg_count
            bg_imgs = list_images(background_dir)
            expected = max(1, len(bg_imgs) * int(per_bg_count))

            # Start a watcher thread to update progress by counting images under output_dir
            stop_flag = threading.Event()

            def watcher():
                last = 0
                while not stop_flag.is_set():
                    cur = len(list_images(output_dir))
                    last = cur
                    st.session_state.progress = min(0.99, cur / max(1, expected))
                    time.sleep(1)

            wt = threading.Thread(target=watcher, daemon=True)
            wt.start()

            # Call backend
            fn(**params)  # blocking call until finished

            stop_flag.set()
            wt.join(timeout=2)
            st.session_state.progress = 1.0

            # Save history record
            rec = BuildRecord(
                run_id=uuid.uuid4().hex,
                dataset_name=dataset_name,
                background_dir=background_dir,
                material_sets=selected_sets,
                rule_file=rule_file,
                created_at=now_str(),
                params={"per_bg_count": per_bg_count, **json.loads(extra_params or "{}")},
                output_dir=str(output_dir),
            )
            HISTORY.append(rec)
            st.success("合成完成并已写入历史记录。")
        except Exception as e:
            st.error(f"运行失败：{e}")
        finally:
            st.session_state.busy = False

    if run_button and not st.session_state.busy:
        threading.Thread(target=run_job, daemon=True).start()

    st.progress(st.session_state.progress)

    # History table
    st.markdown("---")
    st.subheader("📚 历史记录")
    recs = HISTORY.read_all()
    if recs:
        rows = [asdict(r) for r in recs]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("暂无历史记录。")

# -----------------------------------------
# 4) 画廊
# -----------------------------------------
with gallery_tab:
    st.subheader("🖼️ 画廊：查看历史数据集")

    recs = HISTORY.read_all()
    if not recs:
        st.info("还没有构建过数据集。请先去执行一次合成任务。")
    else:
        names = [f"{r.dataset_name}  —  {r.created_at}" for r in recs]
        idx = st.selectbox("选择一个数据集", options=list(range(len(recs))), format_func=lambda i: names[i])
        rec = recs[idx]

        # Preview controls
        show_overlays = st.checkbox("叠加标注（自动识别 YOLO bbox/seg 格式）", value=True)
        cols = st.columns([1, 1])

        out_dir = Path(rec.output_dir)
        aug_imgs = list_images(out_dir)
        if not aug_imgs:
            st.warning("该数据集的输出目录中未找到图片。")
        else:
            # Simple pager
            i = st.number_input("索引", min_value=0, max_value=len(aug_imgs)-1, value=0)
            img_path = aug_imgs[i]

            with cols[0]:
                st.caption("合成结果（右侧可显示叠加标注）")
                st.image(str(img_path), use_column_width=True)

            with cols[1]:
                if show_overlays:
                    st.caption("合成结果 + 标注叠加")
                    st.image(overlay_annotations(img_path), use_column_width=True)
                else:
                    st.caption("仅显示图片")
                    st.image(str(img_path), use_column_width=True)

            st.write(f"路径：{img_path}")
            st.write(f"背景目录：{rec.background_dir}")
            st.write(f"规则：{rec.rule_file}")

# =============================
# --- Annotation Overlay ------
# =============================

def overlay_annotations(img_path: Path | str) -> Image.Image:
    """Try to overlay YOLO-style labels if present.

    We try several common patterns:
    - labels/<stem>.txt or same_dir/<stem>.txt
      Each line either:
        * bbox: cls cx cy w h  (normalized)
        * seg:  cls x1 y1 x2 y2 ... (normalized polygon)
    """
    img_path = Path(img_path)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    cand = [
        img_path.with_suffix(".txt"),
        img_path.parent / "labels" / (img_path.stem + ".txt"),
    ]
    label_path = None
    for c in cand:
        if c.exists():
            label_path = c
            break

    if not label_path:
        return img

    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return img

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            # bbox cx cy w h (normalized)
            cx, cy, w, h = vals
            x1 = (cx - w / 2.0) * W
            y1 = (cy - h / 2.0) * H
            x2 = (cx + w / 2.0) * W
            y2 = (cy + h / 2.0) * H
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=2)
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            # polygon (x1 y1 x2 y2 ... normalized)
            pts = []
            for j in range(0, len(vals), 2):
                x = vals[j] * W
                y = vals[j + 1] * H
                pts.append((x, y))
            draw.polygon(pts, outline=(0, 255, 0, 200))
        else:
            # not recognized
            continue
    return img

# =============================
# ------------- END -----------
# =============================
