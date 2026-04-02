import os
import shutil
import json
import csv
import pandas as pd
import mdai
from glob import glob
from html import escape


class MDAIExporter:
    def __init__(self, config_path, output_dir=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.output_dir = output_dir or self.config.get("output_dir", "mdai_output")
        self.project_id = self.config["mdai_project_id"]
        self.dataset_id = self.config["mdai_dataset_id"]

        self.annotation_vars = self.config.get("annotation_vars", [])
        self.mandatory_annotation_vars = self.config.get(
            "mandatory_annotation_vars", ["labelGroupId", "createdByName"]
        )
        self.dicom_vars = self.config.get("dicom_vars", [])
        self.annotation_filtering = self.config.get("annotation_filtering", True)
        self.dicom_filtering = self.config.get("dicom_filtering", True)

        self.client = mdai.Client(
            domain=self.config.get("mdai_domain", "md.ai"),
            access_token=self.config["mdai_token"]
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------
    # Internal Helpers
    # ----------------------------

    def _find_latest_json(self, hint):
        pattern = os.path.join(self.output_dir, f"*{self.project_id}*{hint}*.json")
        files = glob(pattern)
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]

    def _flatten_entry(self, entry, prefix=""):
        row = {}
        for k, v in entry.items():
            key_name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                row.update(self._flatten_entry(v, key_name))
            else:
                row[key_name] = v if v is not None else ""
        return row

    def _build_user_map(self):
        """Fetch live user map from MD.ai, fall back to config user_map."""
        try:
            users = self.client.project_users(self.project_id)
            user_map = {u["id"]: u.get("name", "Unknown User") for u in users}
            print(f"👤 Loaded {len(user_map)} users from MD.ai")
            return user_map
        except Exception as e:
            print(f"⚠️  Could not fetch project users from MD.ai: {e}")
            print("   Falling back to config user_map.")
            return self.config.get("user_map", {})

    def _copy_json_to_output(self, src_path, label):
        """Copy a JSON file into the output directory and print confirmation."""
        if not src_path or not os.path.exists(src_path):
            return
        dest = os.path.join(self.output_dir, os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(dest):
            shutil.copy2(src_path, dest)
        print(f"📄 {label} JSON saved → {dest}")

    def save_csv(self, rows, filename, variables):
        path = os.path.join(self.output_dir, filename)
        rows = rows or []
        variables_with_sno = ["S.NO"] + variables
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=variables_with_sno)
            writer.writeheader()
            for i, row in enumerate(rows, start=1):
                writer.writerow({"S.NO": i, **{k: row.get(k, "") for k in variables}})

    def save_html(self, rows, filename, variables, title="Table"):
        path = os.path.join(self.output_dir, filename)
        rows = rows or []
        variables_with_sno = ["S.NO"] + variables
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title>")
            f.write("<style>table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:5px;} th{background:#f0f0f0;}</style>")
            f.write(f"</head><body><h2>{title}</h2><table><tr>")
            for var in variables_with_sno:
                f.write(f"<th>{escape(str(var))}</th>")
            f.write("</tr>")
            for i, row in enumerate(rows, start=1):
                f.write("<tr>")
                row_with_sno = {"S.NO": i, **row}
                for var in variables_with_sno:
                    value = escape(str(row_with_sno.get(var, ""))).replace("\n", "<br>")
                    f.write(f"<td>{value}</td>")
                f.write("</tr>")
            f.write("</table></body></html>")

    # ----------------------------
    # Main Workflow Steps
    # ----------------------------

    def download_data(self):
        print(f"🔽 Downloading from MD.ai  Project: {self.project_id} | Dataset: {self.dataset_id} ...")

        label_group_id = self.config.get("mdai_label_group_id")
        kwargs = {
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "path": self.output_dir,
            "annotations_only": True,
        }
        if label_group_id:
            kwargs["label_group_id"] = label_group_id

        self.client.project(**kwargs)

        self.client.download_dicom_metadata(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            format="json",
            path=self.output_dir,
        )
        print("✅ Download complete.")

    def process_annotations(self):
        print("\n📊 Processing Annotations...")

        annotation_file = self._find_latest_json("annotations")
        if not annotation_file:
            print("⚠️  No annotation JSON found — skipping.")
            return

        # Copy raw JSON into output
        self._copy_json_to_output(annotation_file, "Annotations")

        results = mdai.common_utils.json_to_dataframe(annotation_file)
        anno_df = results["annotations"]
        studies_df = results["studies"]
        labels_df = results["labels"]

        # --- Ensure expected label columns exist ---
        for col in ["labelId", "labelName", "labelGroupId", "labelGroupName",
                    "color", "annotationMode", "scope"]:
            if col not in labels_df.columns:
                labels_df[col] = ""
        labels_df = labels_df.fillna("")

        # --- Merge annotations + studies ---
        merged_df = pd.DataFrame()
        if not anno_df.empty and not studies_df.empty:
            studies_df = studies_df.rename(columns={"studyUid": "StudyInstanceUID"})
            merged_df = pd.merge(anno_df, studies_df, on="StudyInstanceUID", how="left")
            merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_y")]
            merged_df.columns = merged_df.columns.str.replace("_x$", "", regex=True)
        elif not anno_df.empty:
            merged_df = anno_df.copy()

        if not merged_df.empty:
            # --- Merge label metadata ---
            if not labels_df.empty:
                label_meta = labels_df[["labelId", "labelName", "color"]].drop_duplicates("labelId")
                merged_df = merged_df.merge(label_meta, on="labelId", how="left")

            # --- Map user IDs → names (live API first, config fallback) ---
            user_map = self._build_user_map()
            if "createdById" in merged_df.columns:
                merged_df["createdByName"] = merged_df["createdById"].map(user_map).fillna("Unknown User")

            # --- Flatten coordinate 'data' column ---
            if "data" in merged_df.columns:
                coords_df = pd.json_normalize(merged_df["data"]).add_prefix("data.")
                merged_df = pd.concat([merged_df.drop(columns=["data"]), coords_df], axis=1)

        merged_df = merged_df.fillna("")

        # --- Annotation filtering ---
        vars_to_keep = list(set(self.annotation_vars + self.mandatory_annotation_vars))
        for col in vars_to_keep:
            if col not in merged_df.columns:
                merged_df[col] = ""

        if self.annotation_filtering and vars_to_keep:
            merged_filtered = merged_df[vars_to_keep]
        else:
            merged_filtered = merged_df

        # --- Build combined labels + annotations rows ---
        combined_rows = []
        if not labels_df.empty:
            for _, row in labels_df.iterrows():
                r = row.to_dict()
                r["Type"] = "Label"
                combined_rows.append(r)
        if not merged_filtered.empty:
            for _, row in merged_filtered.iterrows():
                r = row.to_dict()
                r["Type"] = "Annotation"
                combined_rows.append(r)

        # Determine unified column set
        all_vars = set()
        for r in combined_rows:
            all_vars.update(r.keys())
        all_vars = ["Type"] + sorted(v for v in all_vars if v != "Type")

        # --- Save combined labels + annotations ---
        self.save_csv(combined_rows, f"{self.project_id}_labels_annotations.csv", all_vars)
        self.save_html(combined_rows, f"{self.project_id}_labels_annotations.html", all_vars, "Labels + Annotations")
        print(f"   → Saved: {self.project_id}_labels_annotations.csv / .html")

        # --- Also save annotations-only export (full merged detail) ---
        anno_rows = merged_df.to_dict(orient="records")
        anno_vars = list(merged_df.columns)
        self.save_csv(anno_rows, f"{self.project_id}_annotations.csv", anno_vars)
        self.save_html(anno_rows, f"{self.project_id}_annotations.html", anno_vars, "Annotations")
        print(f"   → Saved: {self.project_id}_annotations.csv / .html")

    def process_dicom(self):
        print("\n🔹 Processing DICOM Metadata...")

        dicom_file = self._find_latest_json("dicom_metadata")
        if not dicom_file:
            print("⚠️  No DICOM metadata JSON found — skipping.")
            return

        # Copy raw JSON into output
        self._copy_json_to_output(dicom_file, "DICOM metadata")

        with open(dicom_file, "r", encoding="utf-8") as f:
            dicoms_data = json.load(f)

        dicom_entries = []
        for dataset in dicoms_data.get("datasets", []):
            dataset_id_val = dataset.get("id", self.config.get("mdai_dataset_id", ""))
            for entry in dataset.get("dicomMetadata", []):
                entry["datasetId"] = dataset_id_val
                dicom_entries.append(entry)

        print(f"   Total DICOM entries loaded: {len(dicom_entries)}")

        flattened = [self._flatten_entry(d) for d in dicom_entries]

        if self.dicom_filtering and self.dicom_vars:
            filtered = [{k: v for k, v in d.items() if k in self.dicom_vars} for d in flattened]
            vars_to_use = self.dicom_vars
        else:
            filtered = flattened
            vars_to_use = sorted({k for d in flattened for k in d})

        self.save_csv(filtered, f"{self.project_id}_dicom.csv", vars_to_use)
        self.save_html(filtered, f"{self.project_id}_dicom.html", vars_to_use, "DICOM Metadata")
        print(f"   → Saved: {self.project_id}_dicom.csv / .html")

    def run(self):
        self.download_data()
        self.process_annotations()
        self.process_dicom()
        print(f"\n✅ All exports complete → {self.output_dir}")


# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    exporter = MDAIExporter(config_path="config.json", output_dir="mdai_output")
    exporter.run()
