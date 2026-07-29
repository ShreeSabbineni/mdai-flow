import os
import json
import pandas as pd
import pydicom
import numpy as np
import cv2
from glob import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sv_ttk  
from mdai.visualize import load_dicom_image

class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD Ai explorer")
        self.root.geometry("1400x820") 
        
        sv_ttk.set_theme("dark")
        
        # State Management
        self.image_cache = {}
        self.current_full_res_img = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        try:
            self.load_config_and_data(config_path)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Could not start: {e}")
            return
            
        self.setup_ui()

    def hex_to_bgr(self, hex_str):
        if not hex_str or not isinstance(hex_str, str):
            return (0, 255, 255)
        hex_str = hex_str.lstrip('#')
        lv = len(hex_str)
        try:
            rgb = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            return (rgb[2], rgb[1], rgb[0])
        except:
            return (0, 255, 255)

    def load_config_and_data(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.project_dir = config.get("Project_dir", "./ProjectDir")
        self.project_id = config.get("mdai_project_id")
        self.user_map = config.get("user_map", {})

        anno_matches = glob(os.path.join(self.project_dir, f"*{self.project_id}*annotations*.json"))
        meta_matches = glob(os.path.join(self.project_dir, f"*{self.project_id}*metadata*.json"))

        if not anno_matches:
            raise FileNotFoundError("Could not find annotation JSON files.")

        with open(anno_matches[0], 'r') as f:
            anno_data = json.load(f)
            annos = anno_data['datasets'][0].get('annotations', [])
            
            self.label_map = {}
            self.color_map = {}
            for g in anno_data.get('labelGroups', []):
                for l in g.get('labels', []):
                    lid = l['id']
                    self.label_map[lid] = l['name']
                    self.color_map[lid] = self.hex_to_bgr(l.get('color'))

        df_annos = pd.json_normalize(annos)
        
        if 'labelId' in df_annos.columns:
            df_annos['labelName'] = df_annos['labelId'].map(self.label_map)

        # --- SAFELY HANDLE DICOM VS MP4 ---
        # 1. Handle DICOMs if metadata exists
        self.merged_dicom = pd.DataFrame()
        if meta_matches:
            with open(meta_matches[0], 'r') as f:
                meta_data = json.load(f)
                meta = meta_data['datasets'][0].get('dicomMetadata', [])
            df_meta = pd.json_normalize(meta)
            if 'SOPInstanceUID' in df_annos.columns:
                dicom_annos = df_annos[df_annos['SOPInstanceUID'].notna()]
                self.merged_dicom = pd.merge(dicom_annos, df_meta, on="SOPInstanceUID", how="inner")

        # 2. Handle Videos (MP4) - Fixed the KeyError here
        self.merged_video = pd.DataFrame()
        if 'SeriesInstanceUID' in df_annos.columns:
            # Videos usually don't have SOPInstanceUIDs in the annotation row
            if 'SOPInstanceUID' in df_annos.columns:
                self.merged_video = df_annos[df_annos['SOPInstanceUID'].isna() & df_annos['SeriesInstanceUID'].notna()].copy()
            else:
                self.merged_video = df_annos[df_annos['SeriesInstanceUID'].notna()].copy()

        # Combine results
        self.merged_df = pd.concat([self.merged_dicom, self.merged_video], ignore_index=True)
        
        # Map IDs to file paths
        self.file_path_map = {}
        for fp in glob(os.path.join(self.project_dir, "*")):
            base_name = os.path.basename(fp)
            if fp.lower().endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True)
                    self.file_path_map[ds.SOPInstanceUID] = base_name
                except: continue
            elif fp.lower().endswith(".mp4"):
                # Use filename without extension as the key for SeriesInstanceUID
                uid_key = os.path.splitext(base_name)[0]
                self.file_path_map[uid_key] = base_name

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        sidebar = ttk.Frame(self.paned, style="Card.TFrame")
        self.paned.add(sidebar, weight=1)
        ttk.Label(sidebar, text="FEED", font=('Segoe UI Variable', 12, 'bold')).pack(pady=10, padx=10, anchor="w")
        
        control_frame = ttk.Frame(sidebar)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_list)
        
        search_row = ttk.Frame(control_frame)
        search_row.pack(fill=tk.X)
        ttk.Entry(search_row, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_row, text="✕", width=2, command=lambda: self.search_var.set("")).pack(side=tk.RIGHT, padx=(2,0))
        ttk.Button(control_frame, text="RESET VIEW", style="Accent.TButton", command=self.reset_zoom).pack(fill=tk.X, pady=(8, 0))

        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.listbox = tk.Listbox(list_frame, font=('Segoe UI Variable', 10), bg="#252525", fg="#ffffff", borderwidth=0, highlightthickness=0)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_container = ttk.Frame(self.paned)
        self.paned.add(right_container, weight=4)

        self.canvas = tk.Canvas(right_container, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<Button-3>", lambda e: self.reset_zoom()) 

        self.info_frame = ttk.LabelFrame(right_container, text=" ANNOTATION INFO ", padding=8)
        self.info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.meta_fields = {}
        fields = [("Label", 0, 0), ("Collaborator", 0, 1), ("Timestamp", 0, 2),
                  ("Frame", 1, 0), ("Task ID", 1, 1), ("Filename", 1, 2)]
        for label, r, c in fields:
            f = ttk.Frame(self.info_frame)
            f.grid(row=r, column=c, sticky="w", padx=12, pady=2)
            ttk.Label(f, text=label.upper(), font=('Segoe UI', 7, 'bold'), foreground="#888888").pack(anchor="w")
            val = ttk.Label(f, text="---", font=('Consolas', 9))
            val.pack(anchor="w")
            self.meta_fields[label] = val

        self.update_listbox(self.merged_df)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    def filter_list(self, *args):
        query = self.search_var.get().lower()
        filtered = self.merged_df[self.merged_df['labelName'].str.lower().str.contains(query, na=False)]
        self.update_listbox(filtered)

    def update_listbox(self, df):
        self.listbox.delete(0, tk.END)
        self.current_df_view = df 
        for _, row in df.iterrows():
            f_num = row.get('frameNumber')
            f_text = f"F{int(f_num)}" if pd.notna(f_num) else "IMG"
            self.listbox.insert(tk.END, f" {row.get('labelName', 'N/A')} — {f_text}")

    def reset_zoom(self):
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
        self.render_image()

    def get_video_frame(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        # Note: MDAI frame numbers are often 1-indexed
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def on_select(self, event=None):
        selection = self.listbox.curselection()
        if not selection: return
        
        row = self.current_df_view.iloc[selection[0]]
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
        
        # Determine if Video or DICOM
        sop_uid = row.get('SOPInstanceUID')
        series_uid = row.get('SeriesInstanceUID')
        
        is_video = pd.isna(sop_uid) and pd.notna(series_uid)
        source_id = series_uid if is_video else sop_uid
        fname = self.file_path_map.get(source_id)
        
        if not fname: 
            messagebox.showwarning("File Missing", f"Could not find file for ID: {source_id}")
            return

        full_path = os.path.join(self.project_dir, fname)
        f_num = int(row.get('frameNumber', 1))

        if is_video:
            img_bgr = self.get_video_frame(full_path, f_num)
            if img_bgr is None: return
        else:
            if source_id not in self.image_cache:
                self.image_cache[source_id] = load_dicom_image(full_path)
            data = self.image_cache[source_id]
            img = data[f_num - 1] if data.ndim > 2 else data
            img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8

        # --- DRAWING ---
        draw_color = self.color_map.get(row.get('labelId'), (0, 255, 255)) 
        if 'data.foreground' in row and isinstance(row['data.foreground'], list):
            for poly in row['data.foreground']:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], True, draw_color, 2, lineType=cv2.LINE_AA)
                overlay = img_bgr.copy()
                cv2.fillPoly(overlay, [pts], draw_color)
                cv2.addWeighted(overlay, 0.15, img_bgr, 0.85, 0, img_bgr)
        elif 'data.x' in row and not pd.isna(row['data.x']):
            cv2.circle(img_bgr, (int(row['data.x']), int(row['data.y'])), 6, draw_color, -1)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.current_full_res_img = Image.fromarray(img_rgb)
        
        # Update Info Panel
        hex_color = '#%02x%02x%02x' % (draw_color[2], draw_color[1], draw_color[0])
        self.meta_fields["Label"].config(text=row.get('labelName', 'Unknown'), foreground=hex_color)
        self.meta_fields["Collaborator"].config(text=self.user_map.get(row.get('createdById'), "N/A"))
        self.meta_fields["Filename"].config(text=fname)
        self.meta_fields["Frame"].config(text=f"Frame {f_num}")
        
        self.render_image()

    def render_image(self):
        if self.current_full_res_img is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10: canvas_w, canvas_h = 800, 500
        width, height = self.current_full_res_img.size
        scale = (min(canvas_w / width, canvas_h / height) * 0.95) * self.zoom_level
        new_size = (int(width * scale), int(height * scale))
        resized = self.current_full_res_img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2 + self.pan_x, canvas_h//2 + self.pan_y, image=self.tk_img)

    def handle_zoom(self, event):
        if event.delta > 0 or event.num == 4: self.zoom_level *= 1.1
        else: self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(self.zoom_level, 8.0))
        self.render_image()

    def start_pan(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def do_pan(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.render_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = MDAIExplorer(root, "configLocal.json") 
    root.mainloop()
