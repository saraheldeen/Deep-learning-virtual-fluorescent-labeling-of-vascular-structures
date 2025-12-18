{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Virtual fluorescent labeling of engineered vascular networks with embedded tracer particles\
\
This repository implements 3D deep-learning models that predict **virtual fluorescence** from **label-free transmission z-stacks** acquired in fibrin vascular constructs containing **2 \'b5m tracer microbeads**. The models generate three fluorescence targets:\
\
- **DAPI** (nuclei)\
- **Phalloidin** (actin / cytoskeleton)\
- **UEA I** (endothelial vasculature)\
\
The intended use is longitudinal imaging workflows where fixation/staining would otherwise prevent repeated measurements and where bead-based microrheology requires keeping tracer particles in the sample.\
\
---\
\
## Figures\
\
### Graphical abstract\
![Graphical Abstract](assets/Graphical_abstract.jpeg)\
\
### Model and pipeline schematic\
![Architecture](assets/Architecture.jpeg)\
\
---\
\
## Input and outputs\
\
### Input\
- **Transmission (label-free) 3D TIFF stack** (Z \'d7 Y \'d7 X or equivalent ordering as produced by your microscope export)\
\
### Outputs (one model per target)\
- **Virtual DAPI** (single-channel)\
- **Virtual Phalloidin** (single-channel)\
- **Virtual UEA I** (single-channel)\
\
---\
\
## Repository layout\
\
### `assets/`\
Images displayed on the GitHub landing page and referenced in this README:\
- `assets/Graphical_abstract.jpeg`\
- `assets/Architecture.jpeg`\
\
### `Example/`\
Example data demonstrating expected stack formatting: **one TIFF stack per channel**.\
- `Example_transmission_stack.tif` \'97 input stack (transmission)\
- `Example_DAPI_stack.tif` \'97 target stack (DAPI)\
- `Example_Phalloidin_stack.tif` \'97 target stack (phalloidin)\
- `Example_UEAI_stack.tif` \'97 target stack (UEA I)\
\
### `metrics/`\
Metric scripts used to evaluate prediction quality and channel-specific morphometrics.\
\
- `metrics/__init__.py`  \
  Keeps `metrics/` importable as a Python package.\
\
- `metrics/common.py`  \
  Shared utilities for metric scripts (pair discovery, TIFF IO helpers, normalization helpers, z-projection helpers, and reusable plotting/config functions where applicable).\
\
- `metrics/image_quality.py`  \
  Computes image-level similarity metrics on paired GT/pred outputs, including:\
  - MSE / RMSE / MAE\
  - SSIM\
  - PSNR\
  - Pearson correlation\
  - CCC (Lin\'92s concordance correlation coefficient)\
\
- `metrics/line_profile.py`  \
  Extracts ROIs and plots GT vs prediction intensity profiles with optional overlay of the sampled line on images.\
\
- `metrics/phalloidin_orientation.py`  \
  Orientation distribution comparison for phalloidin (GT vs prediction) with aggregated plots and nematic-style summary statistics.\
\
- `metrics/ueal_fp_tiles.py`  \
  UEA I false-positive analysis using:\
  - GT-driven vessel masking\
  - Euclidean tolerance band around vessels\
  - strict precision/recall/F1\
  - tolerant precision (ignores near-vessel predictions)\
  - continuous background excess metrics  \
  Supports per-tile (2\'d72) evaluation and CSV outputs.\
\
- `metrics/ueal_vessel_density.py`  \
  Vessel density comparison (GT vs prediction), supporting either whole-image or 2\'d72 tiles, with configurable thresholding and morphological cleanup.\
\
- `metrics/nuclei_segmentation.py`  \
  DAPI evaluation with segmentation-derived outputs:\
  - saves binary masks as ImageJ-ready TIFFs\
  - writes dashed-outline overlays (GT vs prediction)\
  - computes Dice / IoU over masks\
  - supports batch processing across all paired images\
\
### Root scripts\
- `preprocessing.py`  \
  Prepares training data (normalization + patch/volume handling) for transmission inputs and fluorescence targets.\
\
- `train_phalloidin.py`  \
  Trains a 3D U-Net to predict phalloidin from transmission.\
\
- `train_UEAI.py`  \
  Trains a 3D U-Net to predict UEA I from transmission.\
\
- `train_dapi.py`  \
  Trains a 3D U-Net to predict DAPI from transmission.\
\
---\
\
## Model summary\
\
- Architecture: **3D U-Net** encoder\'96decoder with skip connections and (1,2,2) down/upsampling to preserve the short Z dimension while reducing XY.\
- Training: supervised learning on paired transmission/fluorescence volumes or patches.\
- Channel-specific objectives:\
  - **Phalloidin / UEA I:** structure-preserving loss (edge/structure/correlation components)\
  - **DAPI:** sparsity/imbalance-aware loss (overlap- and correlation-based components)\
\
---\
\
## Dependencies\
\
Common requirements for training and metrics:\
- Python 3.x\
- numpy, scipy, pandas\
- tifffile\
- scikit-image\
- opencv-python\
- matplotlib\
- h5py\
- tensorflow, tensorflow-addons\
- tqdm\
\
---\
\
## Citation\
\
If you use this repository in academic work, cite:\
\
\{\\bf \{Eldeen, S.\}\}, Guerrero Ramirez, M. Lanterman, B., Chang, P., \\& Botvinick, E., (2025).\
*Virtual fluorescent labeling of engineered vascular networks with embedded tracer particles.* Acta Biomaterialia. In-progress.\
}