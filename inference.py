"""SegFormer inference engine and AI4Mars dataset browser."""

from __future__ import annotations

import base64
import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


MAX_OUTPUT_DIM = 1536


@dataclass
class DatasetEntry:
    index: int
    split: str
    edr_path: str
    label_path: str
    filename: str
    labeled_frac: float
    is_flipped: bool
    is_brightened: bool


class SegmentationEngine:
    """Loads the trained SMP SegFormer model and runs confidence-aware inference."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        config_path = self.base_dir / "segformer_best.json"
        weights_path = self.base_dir / "segformer_best.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing model config: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing model weights: {weights_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = int(self.config["size"])
        self.num_classes = int(self.config["num_classes"])
        self.class_names: list[str] = list(self.config["class_names"])
        self.colors: dict[str, list[int]] = dict(self.config["colors"])

        self.model = smp.Segformer(
            encoder_name=self.config["encoder"],
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
        )
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor(self.config["mean"], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(self.config["std"], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

    def get_public_config(self) -> dict[str, Any]:
        """Return model metadata and UI palette configuration."""
        return {
            "version": self.config.get("version"),
            "encoder": self.config.get("encoder"),
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "colors": self.colors,
            "test_miou": self.config.get("test_miou"),
            "test_iou": self.config.get("test_iou"),
            "n_test_images": self.config.get("n_test_images"),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
        }

    @staticmethod
    def _encode_png_base64(pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _fit_dimensions(width: int, height: int, max_dim: int = MAX_OUTPUT_DIM) -> tuple[int, int]:
        if max(width, height) <= max_dim:
            return width, height
        scale = max_dim / max(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))

    def predict(self, raw_image: Image.Image) -> dict[str, Any]:
        """Run segmentation and return image + lossless packed (class, 16-bit confidence) PNG."""
        rgb_image = ImageOps.exif_transpose(raw_image).convert("RGB")
        orig_w, orig_h = rgb_image.size
        out_w, out_h = self._fit_dimensions(orig_w, orig_h)

        display_image = (
            rgb_image if (out_w, out_h) == (orig_w, orig_h) else rgb_image.resize((out_w, out_h), Image.BILINEAR)
        )
        model_input = rgb_image.resize((self.input_size, self.input_size), Image.BILINEAR)

        arr = torch.from_numpy(np.array(model_input, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
        tensor = arr.to(device=self.device, dtype=torch.float32) / 255.0
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(tensor)
            logits_up = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits_up, dim=1)
            max_conf, pred_cls = probs.max(dim=1)

        pred_np = pred_cls.squeeze(0).cpu().numpy().astype(np.uint8)
        conf_np = max_conf.squeeze(0).cpu().numpy()

        # Pack (pred_class, 16-bit confidence) into RGBA PNG with A=255 to avoid browser alpha premultiplication.
        conf_u16 = np.clip(np.rint(conf_np * 65535.0), 0, 65535).astype(np.uint16)
        packed = np.empty((out_h, out_w, 4), dtype=np.uint8)
        packed[..., 0] = pred_np
        packed[..., 1] = (conf_u16 >> 8).astype(np.uint8)
        packed[..., 2] = (conf_u16 & 0xFF).astype(np.uint8)
        packed[..., 3] = 255
        packed_img = Image.fromarray(packed, mode="RGBA")

        return {
            "width": out_w,
            "height": out_h,
            "original_width": orig_w,
            "original_height": orig_h,
            "image_data_url": self._encode_png_base64(display_image),
            "prediction_data_url": self._encode_png_base64(packed_img),
            "mean_confidence": float(conf_np.mean()),
        }


class DatasetBrowser:
    """Reads split metadata from CSV and loads images/labels directly from the dataset zip."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.csv_path = self.base_dir / "sample_split_filtered.csv"
        self.zip_path = self._find_zip_file()
        self.entries: list[DatasetEntry] = []
        self._zip: zipfile.ZipFile | None = None

        if self.csv_path.exists() and self.zip_path is not None:
            self._load_entries()

    def _find_zip_file(self) -> Path | None:
        preferred = self.base_dir / "ai4mars_subset(1).zip"
        if preferred.exists():
            return preferred
        zips = sorted(self.base_dir.glob("*.zip"))
        return zips[0] if zips else None

    def _get_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            if self.zip_path is None:
                raise FileNotFoundError("No dataset zip archive found.")
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def _load_entries(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                edr_path = row["edr"]
                self.entries.append(
                    DatasetEntry(
                        index=idx,
                        split=row["split"],
                        edr_path=edr_path,
                        label_path=row["label"],
                        filename=Path(edr_path).name,
                        labeled_frac=float(row["labeled_frac"]),
                        is_flipped=row["is_flipped"].strip().lower() == "true",
                        is_brightened=row["is_brightened"].strip().lower() == "true",
                    )
                )

    @property
    def available(self) -> bool:
        return len(self.entries) > 0 and self.zip_path is not None

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"all": len(self.entries), "test": 0, "val": 0, "train": 0}
        for e in self.entries:
            counts[e.split] = counts.get(e.split, 0) + 1
        return counts

    def query(self, split: str = "test", search: str = "", offset: int = 0, limit: int = 50) -> dict[str, Any]:
        search_lower = search.strip().lower()
        filtered = [
            e
            for e in self.entries
            if (split == "all" or e.split == split) and (not search_lower or search_lower in e.filename.lower())
        ]
        slice_items = filtered[offset : offset + limit]
        return {
            "total": len(filtered),
            "counts": self.split_counts(),
            "items": [
                {
                    "index": e.index,
                    "split": e.split,
                    "filename": e.filename,
                    "labeled_frac": round(e.labeled_frac, 3),
                    "is_flipped": e.is_flipped,
                    "is_brightened": e.is_brightened,
                }
                for e in slice_items
            ],
        }

    def load_sample(self, index: int, target_size: tuple[int, int] | None = None) -> tuple[DatasetEntry, Image.Image, str]:
        if index < 0 or index >= len(self.entries):
            raise IndexError(f"Sample index {index} out of range.")
        entry = self.entries[index]
        zf = self._get_zip()

        img = Image.open(io.BytesIO(zf.read(entry.edr_path)))
        img.load()

        lbl_img = Image.open(io.BytesIO(zf.read(entry.label_path))).convert("L")
        if target_size is not None and lbl_img.size != target_size:
            lbl_img = lbl_img.resize(target_size, Image.NEAREST)

        # Pack raw class ID (0..3, 255) into RGBA R channel with A=255 so canvas reads exact label IDs.
        lbl_arr = np.array(lbl_img, dtype=np.uint8)
        gt_rgba = np.zeros((lbl_arr.shape[0], lbl_arr.shape[1], 4), dtype=np.uint8)
        gt_rgba[..., 0] = lbl_arr
        gt_rgba[..., 3] = 255
        gt_data_url = SegmentationEngine._encode_png_base64(Image.fromarray(gt_rgba, mode="RGBA"))
        return entry, img, gt_data_url
