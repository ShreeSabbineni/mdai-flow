# MDAI Flow: Medical Imaging Annotation Workflow Tools

## Overview

MDAI Flow is a collection of Python-based tools designed to streamline medical imaging annotation workflows using MD.ai.

The project includes two applications:

1. **MDAI Explorer** — an interactive medical imaging annotation viewer for exploring studies, visualizing annotations, and reviewing metadata.
2. **MDAI Utilities** — an automated data processing pipeline that converts MD.ai annotation and DICOM exports into structured CSV and HTML reports for analysis and quality control.

Together, these tools simplify the process of reviewing medical imaging annotations and preparing datasets for downstream research and machine learning workflows.

---

# MDAI Explorer

## Interactive Annotation Viewer

MDAI Explorer provides an interface for reviewing medical imaging studies and associated annotations.

### Features

- Load medical imaging studies
- Visualize annotation overlays
- Review labeled regions of interest
- Inspect annotation metadata
- Navigate through imaging datasets
- Support medical AI annotation review workflows

## Demo

<!-- Add your annotation viewer video here -->

**Demo video coming soon**

## Screenshots

<!-- Add explorer screenshots here -->

---

# MDAI Utilities

## Annotation and DICOM Data Pipeline

MDAI Utilities automates the extraction, processing, and organization of MD.ai project data.

The pipeline:

- Authenticates with MD.ai
- Downloads annotation exports
- Retrieves DICOM metadata
- Processes annotation structures
- Merges annotation and study information
- Maps user IDs to names
- Extracts annotation coordinates
- Generates structured reports

## Generated Outputs

The pipeline produces:

### CSV Reports

Analysis-ready datasets containing:

- Annotation labels
- Study identifiers
- Series identifiers
- Image identifiers
- Creator information
- Annotation coordinates
- DICOM metadata

### HTML Reports

Human-readable tables for quick review and quality control.

Example outputs include:

- Labels and annotations summary
- Annotation metadata tables
- DICOM metadata reports

<!-- Add utilities HTML screenshot here -->

---

# Technical Workflow

```
              MD.ai Project
                   |
                   |
                   v
      Annotation + DICOM Metadata Export
                   |
                   |
                   v
             MDAI Flow Pipeline
                   |
          -----------------------
          |                     |
          v                     v
     CSV Exports          HTML Reports
```

---

# Technology Stack

## Language

- Python

## Libraries

- MD.ai Python SDK
- Pandas
- NumPy
- PyDICOM
- OpenCV
- Pillow
- Matplotlib

## Data Formats

- JSON
- CSV
- HTML
- DICOM metadata

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ShreeSabbineni/mdai-flow.git
cd mdai-flow
```

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

---

# Configuration

Both applications require MD.ai credentials.

Create a local configuration file containing:

- MD.ai domain
- Access token
- Project ID
- Dataset ID

Example:

```json
{
    "mdai_domain": "public.md.ai",
    "mdai_token": "YOUR_TOKEN",
    "mdai_project_id": "YOUR_PROJECT_ID",
    "mdai_dataset_id": "YOUR_DATASET_ID"
}
```

**Important:** Configuration files containing authentication tokens should never be committed to GitHub.

---

# Running the Applications

## Run MDAI Utilities

```bash
py src/mdai_utilities.py
```

The pipeline downloads project data and generates CSV and HTML reports in the output directory.

Example outputs:

```
mdai_output/
│
├── annotations.csv
├── annotations.html
├── dicom.csv
└── dicom.html
```

---

## Run MDAI Explorer

```bash
py src/mdai_explorer.py
```

The application launches an interactive interface for reviewing medical imaging annotations.

---

# Project Motivation

Medical AI development relies on large, carefully annotated datasets. However, reviewing annotations and preparing imaging data for analysis can be complex and time-consuming.

MDAI Flow was developed to make medical imaging workflows more efficient by:

- Improving annotation review
- Automating dataset extraction
- Creating structured analysis outputs
- Making imaging metadata easier to access

---

# Future Improvements

Future extensions may include:

- Automated annotation quality checks
- Additional DICOM visualization tools
- Machine learning pipeline integration
- Support for larger datasets
- Collaborative annotation review features

---

# Acknowledgments

Developed for medical imaging annotation workflows using MD.ai.