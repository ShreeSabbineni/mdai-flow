import os
import json
from glob import glob

import cv2
import mdai
import numpy as np
import pandas as pd
import pydicom
import tkinter as tk
from pydicom.encaps import generate_frames
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image, ImageTk
from tkinter import ttk
import sv_ttk


class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD.ai Explorer")
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
        self.study_series_map = {}
        self.series_to_root = {}
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
            except Exception:
                continue

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

        if (
            not self.df.empty
            and not df_meta.empty
            and "SOPInstanceUID" in self.df.columns
            and "SOPInstanceUID" in df_meta.columns
        ):
            self.df = pd.merge(
                self.df,
                df_meta,
                on="SOPInstanceUID",
                how="left",
                suffixes=("", "_meta")
            )

            for col in ["StudyInstanceUID", "SeriesInstanceUID"]:
                base = col
                meta = f"{col}_meta"
                if base in self.df.columns and meta in self.df.columns:
                    self.df[base] = self.df[base].combine_first(self.df[meta])
                elif meta in self.df.columns and base not in self.df.columns:
                    self.df[base] = self.df[meta]

            for col in ["StudyInstanceUID_meta", "SeriesInstanceUID_meta"]:
                if col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)

        self.df = self.df.reset_index(drop=True)

        if not self.df.empty and "SeriesInstanceUID" in self.df.columns and "StudyInstanceUID" in self.df.columns:
            cols = [c for c in ["datasetId", "StudyInstanceUID", "SeriesInstanceUID"] if c in self.df.columns]
            for _, r in self.df[cols].dropna().drop_duplicates().iterrows():
                dsid = str(r.get("datasetId"))
                study_uid = str(r.get("StudyInstanceUID"))
                series_uid = str(r.get("SeriesInstanceUID"))
                self.study_series_map[(dsid, series_uid)] = study_uid
                root = self.images_dirs.get(r.get("datasetId"))
                if root:
                    self.series_to_root[(dsid, series_uid)] = root

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

    def resolve_media_path(self, row):
        sop = row.get("SOPInstanceUID")
        ser = row.get("SeriesInstanceUID")
        study = row.get("StudyInstanceUID")
        ds_id = row.get("datasetId")

        sop_str = str(sop) if pd.notna(sop) else None
        ser_str = str(ser) if pd.notna(ser) else None
        ds_str = str(ds_id) if pd.notna(ds_id) else None
        key = sop_str if sop_str is not None else ser_str

        if key in self.path_cache:
            return self.path_cache[key]

        root = self.series_to_root.get((ds_str, ser_str), self.images_dirs.get(ds_id))
        if root and ds_str is not None and ser_str is not None:
            self.series_to_root[(ds_str, ser_str)] = root

        if (study is None or pd.isna(study)) and ds_str is not None and ser_str is not None:
            recovered_study = self.study_series_map.get((ds_str, ser_str))
            if recovered_study:
                study = recovered_study

        if sop_str is not None and root and study is not None and not pd.isna(study) and ser_str is not None:
            direct_path = os.path.join(root, str(study), ser_str, f"{sop_str}.dcm")
            if os.path.exists(direct_path):
                self.path_cache[key] = direct_path
                return direct_path

        if sop_str is None and ser_str is not None and root:
            candidate_mp4 = os.path.join(root, f"{ser_str}.mp4")
            if os.path.exists(candidate_mp4):
                self.path_cache[key] = candidate_mp4
                return candidate_mp4

        search_roots = []
        if root:
            search_roots.append(root)
        if self.cache_dir and self.cache_dir != root:
            search_roots.append(self.cache_dir)

        patterns = [f"{sop_str}.dcm", f"{sop_str}*.dcm"] if sop_str is not None else [f"{ser_str}.mp4", f"{ser_str}*.mp4"]

        for search_root in search_roots:
            if not search_root or not os.path.exists(search_root):
                continue
            for pattern in patterns:
                found = glob(os.path.join(search_root, "**", pattern), recursive=True)
                if found:
                    found_path = found[0]
                    self.path_cache[key] = found_path

                    if sop_str is not None:
                        try:
                            learned_series = os.path.basename(os.path.dirname(found_path))
                            learned_study = os.path.basename(os.path.dirname(os.path.dirname(found_path)))
                            learned_root = os.path.dirname(os.path.dirname(os.path.dirname(found_path)))

                            if ds_str is not None:
                                self.images_dirs[ds_id] = learned_root
                                self.study_series_map[(ds_str, learned_series)] = learned_study
                                self.series_to_root[(ds_str, learned_series)] = learned_root

                                if ser_str is not None:
                                    self.study_series_map[(ds_str, ser_str)] = learned_study
                                    self.series_to_root[(ds_str, ser_str)] = learned_root
                        except Exception:
                            pass

                    return found_path

        return None

    def dicom_to_bgr(self, path, frame_idx):
        ds = pydicom.dcmread(path)

        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        frames = int(getattr(ds, "NumberOfFrames", 1))
        photometric = str(getattr(ds, "PhotometricInterpretation", "UNKNOWN"))
        is_encapsulated = bool(getattr(transfer_syntax, "is_encapsulated", False))

        img = None

        if is_encapsulated and frames > 1:
            try:
                frame_bytes = None
                for i, payload in enumerate(generate_frames(ds.PixelData, number_of_frames=frames)):
                    if i == frame_idx:
                        frame_bytes = payload
                        break

                if frame_bytes is None:
                    raise ValueError("Frame not found")

                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("cv2.imdecode failed")
            except Exception:
                img = None

        if img is None:
            pix = ds.pixel_array
            try:
                pix = apply_voi_lut(pix, ds)
            except Exception:
                pass

            img = np.asarray(pix)
            if img.ndim == 4:
                img = img[frame_idx]
            elif img.ndim == 3 and img.shape[0] > 4:
                img = img[frame_idx]

        if img.ndim == 2:
            if photometric == "MONOCHROME1":
                img = img.max() - img

            img = img.astype(np.float32)
            img -= img.min()
            if img.max() > 0:
                img /= img.max()
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img -= img.min()
            if img.max() > 0:
                img /= img.max()
            img = (img * 255).astype(np.uint8)

        if img.ndim == 3 and img.shape[-1] == 3:
            if is_encapsulated and frames > 1:
                return img
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.ndim == 3 and img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def get_video_frame(self, path, frame_idx):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def draw_overlay(self, img_bgr, row):
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

        return img_bgr

    def update_metadata(self, row, path, frame_num):
        self.meta_fields["Label"].config(text=row.get("labelName", "N/A"))
        self.meta_fields["Collaborator"].config(text=self.user_map.get(row.get("createdById"), "N/A"))
        self.meta_fields["Timestamp"].config(text=str(row.get("createdAt", row.get("updatedAt", "---"))))
        self.meta_fields["Frame"].config(text=f"Frame {frame_num}")
        self.meta_fields["Task ID"].config(text=str(row.get("taskId", "---")))
        self.meta_fields["Filename"].config(text=os.path.basename(path))

    def on_select(self, _event):
        if not self.tree.selection():
            return

        row = self.df.iloc[int(self.tree.selection()[0])]
        frame_num = int(row.get("frameNumber", 1)) if pd.notna(row.get("frameNumber")) else 1
        frame_idx = max(0, frame_num - 1)

        path = self.resolve_media_path(row)
        if not path:
            return

        cache_key = (path, frame_idx)
        if cache_key in self.image_cache:
            img_bgr = self.image_cache[cache_key].copy()
        else:
            if path.lower().endswith(".mp4"):
                img_bgr = self.get_video_frame(path, frame_idx)
            else:
                img_bgr = self.dicom_to_bgr(path, frame_idx)

            if img_bgr is None:
                return

            self.image_cache[cache_key] = img_bgr.copy()

        img_bgr = self.draw_overlay(img_bgr, row)
        self.update_metadata(row, path, frame_num)

        self.current_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.render()

    def render(self):
        if self.current_img is None:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        iw, ih = self.current_img.size
        scale = min(cw / iw, ch / ih) * self.zoom_level
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))

        resized = self.current_img.resize(new_size, Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2 + self.pan_x, ch // 2 + self.pan_y, image=self.tk_img)

    def handle_zoom(self, event):
        self.zoom_level *= 1.1 if event.delta > 0 else 0.9
        self.zoom_level = max(0.1, min(self.zoom_level, 8.0))
        if self.current_img is not None:
            self.render()

    def start_pan(self, event):
        self.lx, self.ly = event.x, event.y

    def do_pan(self, event):
        self.pan_x += event.x - self.lx
        self.pan_y += event.y - self.ly
        self.lx, self.ly = event.x, event.y
        if self.current_img is not None:
            self.render()


if __name__ == "__main__":
    root = tk.Tk()
    app = MDAIExplorer(root, "configLocal.json")
    root.mainloop()
