import os
import json
import time
import logging
from glob import glob

import cv2
import mdai
import numpy as np
import pandas as pd
import pydicom
import tkinter as tk
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image, ImageTk
from tkinter import ttk
import sv_ttk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("explorer_debug.log"),
        logging.StreamHandler()
    ]
)

logging.info("pydicom handlers: %s", pydicom.config.pixel_data_handlers)


class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD.ai Explorer FINAL")
        self.root.geometry("1450x850")
        sv_ttk.set_theme("dark")

        self.current_img = None
        self.tk_img = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.lx = 0
        self.ly = 0

        self.cache_dir = r"c:\mdai_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.image_cache = {}
        self.path_cache = {}
        self.images_dirs = {}
        self.user_map = {}
        self.active_filters = {"Label": True, "User": True, "Dataset": True}

        self.load_config_and_data(config_path)
        self.setup_ui()

    def hex_to_bgr(self, hex_str):
        try:
            if not hex_str or not isinstance(hex_str, str):
                return (0, 255, 255)
            hex_str = hex_str.lstrip("#")
            rgb = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])
        except Exception:
            return (0, 255, 255)

    def load_config_and_data(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.project_id = config.get("mdai_project_id")
        self.user_map = config.get("user_map", {})
        dataset_ids = config.get("mdai_dataset_id", [])
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]

        client = mdai.Client(
            domain=config.get("mdai_domain"),
            access_token=config.get("mdai_token")
        )

        all_annos = []
        label_defs = pd.DataFrame()
        all_metas = []

        for d_id in dataset_ids:
            try:
                project = client.project(self.project_id, dataset_id=d_id, path=self.cache_dir)
                if project and hasattr(project, "images_dir") and project.images_dir:
                    self.images_dirs[d_id] = project.images_dir

                client.project(self.project_id, dataset_id=d_id, annotations_only=True)
                client.download_dicom_metadata(self.project_id, d_id, path=self.cache_dir)

                anno_files = glob(os.path.join(self.cache_dir, f"*annotations*{d_id}*.json"))
                if anno_files:
                    latest_anno = max(anno_files, key=os.path.getmtime)
                    res = mdai.common_utils.json_to_dataframe(latest_anno)

                    if res.get("annotations") is not None:
                        df_anno = res["annotations"].copy()
                        df_anno["datasetId"] = d_id
                        all_annos.append(df_anno)

                    if res.get("labels") is not None:
                        label_defs = pd.concat([label_defs, res["labels"]], ignore_index=True)
                        label_defs = label_defs.drop_duplicates(subset=["labelId"])

                meta_files = glob(os.path.join(self.cache_dir, f"*metadata*{d_id}*.json"))
                if meta_files:
                    latest_meta = max(meta_files, key=os.path.getmtime)
                    with open(latest_meta, "r", encoding="utf-8") as mf:
                        meta_json = json.load(mf)
                    if "datasets" in meta_json:
                        df_meta = pd.json_normalize(meta_json["datasets"][0].get("dicomMetadata", []))
                        all_metas.append(df_meta)
            except Exception as e:
                logging.exception("Dataset load failed for %s: %s", d_id, e)

        self.df = pd.concat(all_annos, ignore_index=True) if all_annos else pd.DataFrame()
        df_meta = pd.concat(all_metas, ignore_index=True) if all_metas else pd.DataFrame()

        self.label_map = (
            dict(zip(label_defs["labelId"], label_defs["labelName"]))
            if not label_defs.empty and "labelId" in label_defs and "labelName" in label_defs
            else {}
        )
        self.color_map = (
            {row["labelId"]: self.hex_to_bgr(row.get("color")) for _, row in label_defs.iterrows()}
            if not label_defs.empty and "labelId" in label_defs
            else {}
        )

        if not self.df.empty:
            self.df["labelName"] = self.df["labelId"].map(self.label_map)
            if "data" in self.df.columns:
                coords = pd.json_normalize(self.df["data"]).add_prefix("data.")
                self.df = pd.concat([self.df.drop(columns=["data"]), coords], axis=1)

        if not self.df.empty and not df_meta.empty and "SOPInstanceUID" in self.df.columns and "SOPInstanceUID" in df_meta.columns:
            self.df = pd.merge(self.df, df_meta, on="SOPInstanceUID", how="left")

        self.df = self.df.reset_index(drop=True)

    def setup_ui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(self.paned, width=480)
        self.paned.add(sidebar, weight=1)

        search_frame = ttk.Frame(sidebar)
        search_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_table)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(fill=tk.X)

        btn_frame = ttk.Frame(sidebar)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.btns = {}
        for name in ["Label", "User", "Dataset"]:
            btn = tk.Button(
                btn_frame,
                text=f"● {name}",
                command=lambda x=name: self.toggle_filter(x),
                bg="#005fb8",
                fg="white",
                relief="flat",
                font=("Segoe UI", 8, "bold"),
                pady=5,
            )
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            self.btns[name] = btn

        cols = ("#", "Label", "User", "Dataset")
        self.tree = ttk.Treeview(sidebar, columns=cols, show="", selectmode="browse")
        self.tree.column("#", width=40, anchor="center")
        self.tree.column("Label", width=160, anchor="w")
        self.tree.column("User", width=130, anchor="w")
        self.tree.column("Dataset", width=120, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        viewer = ttk.Frame(self.paned)
        self.paned.add(viewer, weight=4)

        self.canvas = tk.Canvas(viewer, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info_frame = ttk.LabelFrame(viewer, text=" ANNOTATION INFO ", padding=8)
        self.info_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.meta_fields = {}
        fields = [
            ("Label", 0, 0),
            ("Collaborator", 0, 1),
            ("Timestamp", 0, 2),
            ("Frame", 1, 0),
            ("Task ID", 1, 1),
            ("Filename", 1, 2),
        ]
        for label, r, c in fields:
            frame = ttk.Frame(self.info_frame)
            frame.grid(row=r, column=c, sticky="w", padx=15, pady=2)
            ttk.Label(frame, text=label.upper(), font=("Segoe UI", 7, "bold"), foreground="#888888").pack(anchor="w")
            val = ttk.Label(frame, text="---", font=("Consolas", 9))
            val.pack(anchor="w")
            self.meta_fields[label] = val

        self.debug_text = tk.Text(viewer, height=10, bg="#111111", fg="#dddddd", insertbackground="#dddddd")
        self.debug_text.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

        self.filter_table()

    def toggle_filter(self, name):
        self.active_filters[name] = not self.active_filters[name]
        self.filter_table()

    def filter_table(self, *args):
        query = self.search_var.get().lower()

        for item in self.tree.get_children():
            self.tree.delete(item)

        visible_labels = set()
        visible_users = set()
        visible_datasets = set()

        row_num = 1
        for idx, row in self.df.iterrows():
            label = str(row.get("labelName", ""))
            created_by = row.get("createdById")
            user = self.user_map.get(created_by, created_by if created_by else "Unknown")
            dataset = str(row.get("datasetId", "N/A"))

            match = query == ""
            if not match:
                if self.active_filters["Label"] and query in label.lower():
                    match = True
                if self.active_filters["User"] and query in str(user).lower():
                    match = True
                if self.active_filters["Dataset"] and query in dataset.lower():
                    match = True

            if match:
                self.tree.insert("", tk.END, iid=str(idx), values=(row_num, label, user, dataset))
                visible_labels.add(label)
                visible_users.add(user)
                visible_datasets.add(dataset)
                row_num += 1

        counts = {
            "Label": len(visible_labels),
            "User": len(visible_users),
            "Dataset": len(visible_datasets),
        }
        for name in ["Label", "User", "Dataset"]:
            active = self.active_filters[name]
            self.btns[name].config(
                text=f"{'●' if active else '○'} {name} ({counts[name]})",
                bg="#005fb8" if active else "#333333"
            )

    def resolve_media_path(self, row, timings):
        start = time.perf_counter()

        sop = row.get("SOPInstanceUID")
        ser = row.get("SeriesInstanceUID")
        study = row.get("StudyInstanceUID")
        ds_id = row.get("datasetId")

        key = str(sop) if pd.notna(sop) else str(ser)
        if key in self.path_cache:
            timings["path_cache_hit"] = time.perf_counter() - start
            logging.info("path_cache HIT %s", key)
            return self.path_cache[key]

        root = self.images_dirs.get(ds_id)

        if pd.notna(sop) and root and pd.notna(study) and pd.notna(ser):
            direct_start = time.perf_counter()
            direct_path = os.path.join(root, str(study), str(ser), f"{sop}.dcm")
            if os.path.exists(direct_path):
                self.path_cache[key] = direct_path
                timings["direct_path"] = time.perf_counter() - direct_start
                timings["resolve_media_path_total"] = time.perf_counter() - start
                logging.info("direct path HIT %s", direct_path)
                return direct_path
            timings["direct_path"] = time.perf_counter() - direct_start

        search_roots = [root, self.cache_dir]
        if pd.notna(sop):
            patterns = [f"{sop}.dcm", f"{sop}*.dcm"]
        else:
            patterns = [f"{ser}.mp4", f"{ser}*.mp4"]

        glob_start = time.perf_counter()
        for search_root in search_roots:
            if not search_root or not os.path.exists(search_root):
                continue
            for pattern in patterns:
                found = glob(os.path.join(search_root, "**", pattern), recursive=True)
                if found:
                    self.path_cache[key] = found[0]
                    timings["glob_fallback"] = time.perf_counter() - glob_start
                    timings["resolve_media_path_total"] = time.perf_counter() - start
                    logging.info("fallback HIT %s", found[0])
                    return found[0]

        timings["glob_fallback"] = time.perf_counter() - glob_start
        timings["resolve_media_path_total"] = time.perf_counter() - start
        logging.info("path NOT FOUND for %s", key)
        return None

    def dicom_to_bgr(self, path, frame_idx, timings):
        start = time.perf_counter()

        t = time.perf_counter()
        ds = pydicom.dcmread(path)
        timings["dcmread"] = time.perf_counter() - t

        timings["transfer_syntax"] = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "UNKNOWN"))
        timings["rows"] = int(getattr(ds, "Rows", -1))
        timings["cols"] = int(getattr(ds, "Columns", -1))
        timings["frames"] = int(getattr(ds, "NumberOfFrames", 1))
        timings["samples_per_pixel"] = int(getattr(ds, "SamplesPerPixel", 1))
        timings["bits_allocated"] = int(getattr(ds, "BitsAllocated", -1))
        timings["photometric"] = str(getattr(ds, "PhotometricInterpretation", "UNKNOWN"))

        t = time.perf_counter()
        pix = ds.pixel_array
        timings["pixel_array"] = time.perf_counter() - t

        t = time.perf_counter()
        try:
            pix = apply_voi_lut(pix, ds)
        except Exception:
            pass
        timings["apply_voi_lut"] = time.perf_counter() - t

        t = time.perf_counter()
        img = np.asarray(pix)
        if img.ndim == 4:
            img = img[frame_idx]
        elif img.ndim == 3 and img.shape[0] > 4:
            img = img[frame_idx]
        timings["frame_select"] = time.perf_counter() - t

        t = time.perf_counter()
        if img.ndim == 2:
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = img.max() - img

            img = img.astype(np.float32)
            img -= img.min()
            if img.max() > 0:
                img /= img.max()
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img -= img.min()
                if img.max() > 0:
                    img /= img.max()
                img = (img * 255).astype(np.uint8)

            if img.ndim == 3 and img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.ndim == 3 and img.shape[-1] == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        timings["dicom_to_bgr_convert"] = time.perf_counter() - t
        timings["dicom_total"] = time.perf_counter() - start
        return img_bgr

    def get_video_frame(self, path, frame_idx, timings):
        start = time.perf_counter()

        t = time.perf_counter()
        cap = cv2.VideoCapture(path)
        timings["video_open"] = time.perf_counter() - t

        t = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        timings["video_seek"] = time.perf_counter() - t

        t = time.perf_counter()
        ok, frame = cap.read()
        timings["video_read"] = time.perf_counter() - t

        t = time.perf_counter()
        cap.release()
        timings["video_release"] = time.perf_counter() - t
        timings["video_total"] = time.perf_counter() - start

        return frame if ok else None

    def draw_overlay(self, img_bgr, row, timings):
        start = time.perf_counter()
        color = self.color_map.get(row.get("labelId"), (0, 255, 0))

        if "data.foreground" in row and isinstance(row["data.foreground"], list):
            for poly in row["data.foreground"]:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], True, color, 2, lineType=cv2.LINE_AA)
        elif pd.notna(row.get("data.x")) and pd.notna(row.get("data.y")):
            x, y = int(row["data.x"]), int(row["data.y"])
            if pd.notna(row.get("data.width")) and pd.notna(row.get("data.height")):
                w, h = int(row["data.width"]), int(row["data.height"])
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
            else:
                cv2.circle(img_bgr, (x, y), 5, color, -1)

        timings["overlay_draw"] = time.perf_counter() - start
        return img_bgr

    def update_metadata(self, row, path, frame_num, timings):
        start = time.perf_counter()
        self.meta_fields["Label"].config(text=row.get("labelName", "N/A"))
        self.meta_fields["Collaborator"].config(text=self.user_map.get(row.get("createdById"), "N/A"))
        self.meta_fields["Timestamp"].config(text=str(row.get("createdAt", row.get("updatedAt", "---"))))
        self.meta_fields["Frame"].config(text=f"Frame {frame_num}")
        self.meta_fields["Task ID"].config(text=str(row.get("taskId", "---")))
        self.meta_fields["Filename"].config(text=os.path.basename(path))
        timings["metadata_update"] = time.perf_counter() - start

    def on_select(self, _event):
        if not self.tree.selection():
            return

        total_start = time.perf_counter()
        timings = {}

        row = self.df.iloc[int(self.tree.selection()[0])]
        frame_num = int(row.get("frameNumber", 1)) if pd.notna(row.get("frameNumber")) else 1
        frame_idx = max(0, frame_num - 1)

        path = self.resolve_media_path(row, timings)
        if not path:
            self.debug_text.delete("1.0", tk.END)
            self.debug_text.insert(tk.END, "Path not found.\n")
            return

        cache_key = (path, frame_idx)
        cache_start = time.perf_counter()
        if cache_key in self.image_cache:
            img_bgr = self.image_cache[cache_key].copy()
            timings["image_cache_hit"] = time.perf_counter() - cache_start
            logging.info("image_cache HIT %s", cache_key)
        else:
            timings["image_cache_hit"] = 0.0
            if path.lower().endswith(".mp4"):
                img_bgr = self.get_video_frame(path, frame_idx, timings)
            else:
                img_bgr = self.dicom_to_bgr(path, frame_idx, timings)

            if img_bgr is None:
                self.debug_text.delete("1.0", tk.END)
                self.debug_text.insert(tk.END, "Failed to decode media.\n")
                return

            self.image_cache[cache_key] = img_bgr.copy()
            timings["image_cache_store"] = time.perf_counter() - cache_start

        img_bgr = self.draw_overlay(img_bgr, row, timings)
        self.update_metadata(row, path, frame_num, timings)

        t = time.perf_counter()
        self.current_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        timings["bgr_to_pil"] = time.perf_counter() - t

        self.render(timings)
        timings["total_select"] = time.perf_counter() - total_start

        logging.info("TIMINGS %s", timings)
        self.debug_text.delete("1.0", tk.END)
        for k in sorted(timings.keys()):
            v = timings[k]
            if isinstance(v, (int, float)):
                self.debug_text.insert(tk.END, f"{k}: {v:.4f}s\n")
            else:
                self.debug_text.insert(tk.END, f"{k}: {v}\n")

    def render(self, timings):
        if self.current_img is None:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        iw, ih = self.current_img.size
        scale = min(cw / iw, ch / ih) * self.zoom_level
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))

        t = time.perf_counter()
        resized = self.current_img.resize(new_size, Image.BILINEAR)
        timings["pil_resize"] = time.perf_counter() - t

        t = time.perf_counter()
        self.tk_img = ImageTk.PhotoImage(resized)
        timings["photoimage"] = time.perf_counter() - t

        t = time.perf_counter()
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2 + self.pan_x, ch // 2 + self.pan_y, image=self.tk_img)
        timings["canvas_draw"] = time.perf_counter() - t

    def handle_zoom(self, event):
        self.zoom_level *= 1.1 if event.delta > 0 else 0.9
        self.zoom_level = max(0.1, min(self.zoom_level, 8.0))
        if self.current_img is not None:
            self.render({})

    def start_pan(self, event):
        self.lx, self.ly = event.x, event.y

    def do_pan(self, event):
        self.pan_x += event.x - self.lx
        self.pan_y += event.y - self.ly
        self.lx, self.ly = event.x, event.y
        if self.current_img is not None:
            self.render({})


if __name__ == "__main__":
    root = tk.Tk()
    app = MDAIExplorer(root, "configLocal.json")
    root.mainloop()
