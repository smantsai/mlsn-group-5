# AI4Mars SegFormer Studio

A local web application for **confidence-thresholded Mars terrain semantic segmentation** powered by a trained `mit_b2` SegFormer model (`segmentation-models-pytorch`) on the AI4Mars dataset.

Upload any terrain image—or browse images directly from the AI4Mars dataset archive—and interactively adjust the **confidence threshold (`0%–100%`)**: pixels where the model's softmax confidence is at least `X%` are labeled with their predicted terrain class, while lower-confidence pixels are left unlabeled.

## Terrain Classes

| ID | Class | Color | Test IoU |
| --- | --- | --- | --- |
| `0` | `soil` | Red (`255, 0, 0`) | `0.9468` |
| `1` | `bedrock` | Yellow (`255, 255, 0`) | `0.8525` |
| `2` | `sand` | Green (`0, 255, 0`) | `0.9160` |
| `3` | `big rock` | Cyan (`0, 255, 255`) | `0.5632` |

- **Backbone / Model**: `mit_b2` SegFormer (`512×512` input, ImageNet mean/std normalization)
- **Overall Test mIoU**: `0.8196` across `372` evaluation images (`247` test + `125` val)

## Features

- **Instant Confidence Thresholding**: Adjust the threshold slider (`0%–100%`) or presets in real time with zero server re-inference.
- **Upload & Dataset Explorer**: Drag-and-drop any local image (`JPG`, `PNG`, `BMP`, `WEBP`), or search/browse `test`, `val`, and `train` splits directly from `ai4mars_subset.zip` + `sample_split_filtered.csv`.
- **Multiple View Modes**: `Overlay`, `Mask Only`, `Side-by-Side`, `Confidence Map`, and `vs Ground Truth` (with live accuracy and mIoU at the selected threshold).
- **Pixel Inspector & Export**: Hover over any pixel to inspect coordinates, predicted class, confidence `%`, and ground truth; download thresholded overlay or transparent mask PNGs.

## Project Structure

```text
.
├── app.py                      # FastAPI backend server
├── inference.py                # SegFormer inference engine & zip dataset reader
├── segformer_best.json         # Model config, class palette, normalization & metrics
├── segformer_best.pt           # Trained PyTorch state_dict
├── sample_split_filtered.csv   # Train / val / test split index
├── requirements.txt            # Python dependencies
├── run_app.bat                 # Windows one-click launcher
└── static/
    ├── index.html              # Web UI
    ├── styles.css              # Dark UI theme
    └── app.js                  # Client-side thresholding, rendering & stats
```

> **Note on the dataset archive**: `*.zip` files (such as `ai4mars_subset.zip`, ~412 MB) are excluded in `.gitignore` because they exceed GitHub's 100 MB file limit. Image upload and segmentation work without the zip; place `ai4mars_subset.zip` in the project root if you also want to use the built-in Dataset Explorer and Ground Truth comparison.

## Getting Started

### Option 1: Quick Launch (if `.venv` is already set up)

Double-click `run_app.bat`, or run in PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open **http://127.0.0.1:8000** in your browser.

### Option 2: Fresh Setup (new machine / after cloning)

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   *(On macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`)*

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   *(For GPU acceleration on Windows/Linux, install the CUDA-enabled PyTorch wheel matching your drivers from [pytorch.org](https://pytorch.org/get-started/locally/).)*

3. Start the server:

   ```powershell
   python app.py
   ```

4. Open **http://127.0.0.1:8000** in your browser.
