import os
import json
import pandas as pd
import numpy as np
import cv2
import mdai
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import sys
from glob import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sv_ttk

class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD.ai Cloud Explorer")
        self.root.geometry("1450x850")
        sv_ttk.set_theme("dark")

        # State Management
        self.current_full_res_img = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.project_instances = {}
        self.images_dirs = {}   
        self.cache_dir = r"c:\mdai_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.active_filters = {"Label": True, "User": True, "Dataset": True}

        try:
            self.load_config_and_data(config_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            return

        self.setup_ui()

    def hex_to_bgr(self, hex_str):
        if not hex_str or not isinstance(hex_str, str):
            return (0, 255, 255)
        hex_str = hex_str.lstrip('#')
        rgb = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

    def silence(self):
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = sys.stderr = self._devnull

    def unsilence(self):
        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
        self._devnull.close()

    def load_config_and_data(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.project_id = config.get("mdai_project_id")
        self.token = config.get("mdai_token")
        self.domain = config.get("mdai_domain", "ucsf.md.ai")
        self.user_map = config.get("user_map", {})
        d_ids = config.get("mdai_dataset_id", [])
        if isinstance(d_ids, str): d_ids = [d_ids]

        self.silence()
        try:
            client = mdai.Client(domain=self.domain, access_token=self.token)
        finally:
            self.unsilence()

        all_annos, all_metas, label_defs = [], [], pd.DataFrame()

        for d_id in d_ids:
            try:
                self.silence()
                p = client.project(self.project_id, dataset_id=d_id, path=self.cache_dir)
                self.project_instances[d_id] = p
                if p and hasattr(p, 'images_dir'): self.images_dirs[d_id] = p.images_dir
                client.project(self.project_id, dataset_id=d_id, annotations_only=True)
                client.download_dicom_metadata(self.project_id, d_id, path=self.cache_dir)
                self.unsilence()

                anno_files = glob(os.path.join(self.cache_dir, f"*{self.project_id}*annotations*{d_id}*.json"))
                if anno_files:
                    res = mdai.common_utils.json_to_dataframe(max(anno_files, key=os.path.getmtime))
                    if res['annotations'] is not None:
                        # CRITICAL: Attach datasetId here to prevent N/A
                        res['annotations']['datasetId'] = d_id
                        all_annos.append(res['annotations'])
                    if res['labels'] is not None:
                        label_defs = pd.concat([label_defs, res['labels']]).drop_duplicates(subset=['labelId'])

                meta_files = glob(os.path.join(self.cache_dir, f"*{self.project_id}*metadata*{d_id}*.json"))
                if meta_files:
                    with open(max(meta_files, key=os.path.getmtime), 'r', encoding='utf-8') as f:
                        m_json = json.load(f)
                        df_m = pd.json_normalize(m_json['datasets'][0].get('dicomMetadata', []))
                        # Note: we don't strictly need meta datasetId if anno has it, but it helps
                        df_m['datasetId_meta'] = d_id 
                        all_metas.append(df_m)
            except: continue

        df_a = pd.concat(all_annos, ignore_index=True) if all_annos else pd.DataFrame()
        df_m = pd.concat(all_metas, ignore_index=True) if all_metas else pd.DataFrame()
        
        self.label_map = dict(zip(label_defs.labelId, label_defs.labelName)) if not label_defs.empty else {}
        self.color_map = {r['labelId']: self.hex_to_bgr(r.get('color')) for _, r in label_defs.iterrows()} if not label_defs.empty else {}

        if not df_a.empty:
            df_a['labelName'] = df_a['labelId'].map(self.label_map)
            if 'data' in df_a.columns:
                coords = pd.json_normalize(df_a['data']).add_prefix('data.')
                df_a = pd.concat([df_a.drop(columns=['data']), coords], axis=1)

        # Merge on SOPInstanceUID, but keep the datasetId from df_a
        if not df_m.empty:
            self.merged_df = pd.merge(df_a, df_m, on="SOPInstanceUID", how="left")
        else:
            self.merged_df = df_a
            
        self.merged_df = self.merged_df.reset_index(drop=True)

    def setup_ui(self):
        for widget in self.root.winfo_children(): widget.destroy()
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar = ttk.Frame(self.paned, width=480)
        self.paned.add(self.sidebar, weight=1)

        search_frame = ttk.Frame(self.sidebar)
        search_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_table)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(fill=tk.X)

        btn_frame = ttk.Frame(self.sidebar)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.btns = {}
        for f in ["Label", "User", "Dataset"]:
            b = tk.Button(btn_frame, text=f"● {f}", command=lambda x=f: self.toggle_filter(x),
                          bg="#005fb8", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=5)
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            self.btns[f] = b

        # Added "#" column for numbering 1, 2, 3...
        cols = ("#", "Label", "User", "Dataset")
        self.tree = ttk.Treeview(self.sidebar, columns=cols, show="", selectmode="browse")
        self.tree.column("#", width=40, anchor="center")
        self.tree.column("Label", width=140)
        self.tree.column("User", width=120)
        self.tree.column("Dataset", width=120)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_item_select)

        # Viewer Panel
        self.v_frame = ttk.Frame(self.paned)
        self.paned.add(self.v_frame, weight=4)
        self.canvas = tk.Canvas(self.v_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Info Grid
        self.info_frame = ttk.LabelFrame(self.v_frame, text=" ANNOTATION INFO ", padding=8)
        self.info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.meta_fields = {}
        fields = [("Label", 0, 0), ("Collaborator", 0, 1), ("Timestamp", 0, 2),
                  ("Frame", 1, 0), ("Task ID", 1, 1), ("Filename", 1, 2)]
        for label, r, c in fields:
            f = ttk.Frame(self.info_frame)
            f.grid(row=r, column=c, sticky="w", padx=15, pady=2)
            ttk.Label(f, text=label.upper(), font=('Segoe UI', 7, 'bold'), foreground="#888888").pack(anchor="w")
            val = ttk.Label(f, text="---", font=('Consolas', 9))
            val.pack(anchor="w")
            self.meta_fields[label] = val

        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self, 'lx', e.x) or setattr(self, 'ly', e.y))
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.populate_table()

    def toggle_filter(self, f):
        self.active_filters[f] = not self.active_filters[f]
        self.filter_table()

    def filter_table(self, *args):
        q = self.search_var.get().lower()
        for i in self.tree.get_children(): self.tree.delete(i)
        
        visible_count = 0
        unique_labels = set()
        unique_users = set()
        unique_datasets = set()

        # Counter for the 1, 2, 3 indexing
        list_index = 1

        for idx, row in self.merged_df.iterrows():
            lbl = str(row.get('labelName', ''))
            uid = row.get('createdById')
            user = self.user_map.get(uid, uid if uid else "Unknown")
            ds = str(row.get('datasetId', 'N/A'))
            
            # Check if it matches search
            if q == "" or any(q in x.lower() for x in [lbl, user, ds]):
                # Treeview numbering
                self.tree.insert("", tk.END, iid=str(idx), values=(list_index, lbl, user, ds))
                
                # Track data for button counts
                unique_labels.add(lbl)
                unique_users.add(user)
                unique_datasets.add(ds)
                visible_count += 1
                list_index += 1

        # Update Buttons to show "How many match this search"
        # If search is empty, it shows total project counts
        for key, count_val in zip(["Label", "User", "Dataset"], 
                                 [len(unique_labels), len(unique_users), len(unique_datasets)]):
            status = "●" if self.active_filters[key] else "○"
            bg = "#005fb8" if self.active_filters[key] else "#333333"
            self.btns[key].config(text=f"{status} {key} ({count_val})", bg=bg)

    def populate_table(self): self.filter_table()

    def dicom_to_bgr(self, path, frame_idx=0):
        ds = pydicom.dcmread(path)
        pix = ds.pixel_array
        try: pix = apply_voi_lut(pix, ds)
        except: pass
        img = np.asarray(pix)
        if img.ndim == 4: img = img[frame_idx]
        elif img.ndim == 3 and img.shape[0] > 4: img = img[frame_idx]
        
        if img.ndim == 2:
            img = img.astype(np.float32)
            img -= img.min()
            if img.max() > 0: img /= img.max()
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def on_item_select(self, e):
        sel = self.tree.selection()
        if not sel: return
        row = self.merged_df.iloc[int(sel[0])]
        
        f_val = row.get('frameNumber', 1)
        f_num = int(f_val) if pd.notna(f_val) else 1
        
        sop, ser, ds_id = row.get('SOPInstanceUID'), row.get('SeriesInstanceUID'), row.get('datasetId')
        is_vid = pd.isna(sop) and pd.notna(ser)
        sid = ser if is_vid else sop
        
        path = None
        for d in [self.images_dirs.get(ds_id), self.cache_dir]:
            found = glob(os.path.join(d, "**", f"{sid}*"), recursive=True) if d else []
            if found: path = found[0]; break
        if not path: return

        if path.lower().endswith('.mp4'):
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_num - 1))
            ret, img_bgr = cap.read()
            cap.release()
        else:
            img_bgr = self.dicom_to_bgr(path, frame_idx=max(0, f_num-1))

        if img_bgr is None: return

        # Drawing
        color = self.color_map.get(row['labelId'], (0, 255, 0))
        if 'data.foreground' in row and isinstance(row['data.foreground'], list):
            for poly in row['data.foreground']:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], True, color, 2)
        elif 'data.x' in row and pd.notna(row['data.x']):
            x, y = int(row['data.x']), int(row['data.y'])
            if 'data.width' in row and pd.notna(row['data.width']):
                cv2.rectangle(img_bgr, (x, y), (x+int(row['data.width']), y+int(row['data.height'])), color, 2)
            else: cv2.circle(img_bgr, (x, y), 5, color, -1)

        # Bottom UI
        self.meta_fields["Label"].config(text=row.get('labelName', 'N/A'))
        self.meta_fields["Collaborator"].config(text=self.user_map.get(row.get('createdById'), "N/A"))
        self.meta_fields["Frame"].config(text=f"Frame {f_num}")
        self.meta_fields["Filename"].config(text=os.path.basename(path))

        self.current_full_res_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.render_image()

    def render_image(self):
        if not self.current_full_res_img: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        iw, ih = self.current_full_res_img.size
        s = min(cw/iw, ch/ih) * self.zoom_level
        self.tk_img = ImageTk.PhotoImage(self.current_full_res_img.resize((int(iw*s), int(ih*s)), Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(cw//2 + self.pan_x, ch//2 + self.pan_y, image=self.tk_img)

    def handle_zoom(self, e):
        self.zoom_level *= 1.1 if e.delta > 0 else 0.9
        self.render_image()

    def do_pan(self, e):
        self.pan_x += (e.x - self.lx); self.pan_y += (e.y - self.ly)
        self.lx, self.ly = e.x, e.y
        self.render_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = MDAIExplorer(root, "configLocal.json")
    root.mainloop()
