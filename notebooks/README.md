# Colab Training

`colab_train.ipynb` is a self-contained notebook that runs the **full pipeline** end-to-end on a Colab GPU runtime: data preprocessing (mask generation + patches + train/val split) → segmentation training → classification training.

## Quickstart

1. **Get the raw FPIC dataset** (request access codes from the dataset authors — see top-level [README.md](../README.md)).
2. **Zip raw inputs** and upload to Drive:
    ```powershell
    Compress-Archive -Path data\pcb_image, data\smd_annotation -DestinationPath data_raw.zip -Force
    ```
    Place at `MyDrive/PCBSegClassNet/data_raw.zip` (~7 GB).
3. **Open the notebook in Colab**:
    ```
    https://colab.research.google.com/github/<your-fork>/PCBSegClassNet/blob/colab/notebooks/colab_train.ipynb
    ```
4. **Runtime → Change runtime type → GPU** (T4 is enough; High-RAM not needed), then run cells top to bottom.

## What the notebook does

| Section | Purpose |
|---|---|
| 1 | `nvidia-smi` GPU sanity |
| 2 | Clone this repo (`colab` branch) |
| 3 | Install TF 2.15 + dependencies (TF 2.15 is the last release on Keras 2; Keras 3 from TF 2.16+ breaks this codebase's `tf.keras.backend.{dot,transpose}` calls) |
| 4 | Mount Drive, unzip `data_raw.zip` to local Colab disk |
| 5 | `create_mask.py` — polygon masks + classification crops (EDSR super-resolution, GPU) |
| 6 | `create_patches.py` — 768 px patches + 80/20 train/val split (CPU) |
| 7 | Set up Drive checkpoint directory for persistence across sessions |
| 8 | Segmentation training (5 epochs sanity → 80 epochs full → mirror checkpoint to Drive) |
| 9 | Classification training (same pattern) |
| 10 | Optional: re-evaluate from Drive checkpoints in a fresh session |

## Why preprocess on Colab?

- Raw inputs (~7 GB) are smaller than the processed dataset (~18 GB) — easier to transfer to Drive.
- Reproducibility: anyone with raw data + this notebook can recreate the exact training set without trusting an opaque processed zip.
- Easy to iterate on preprocessing knobs (e.g. patch size) without re-uploading.

If you already have a processed dataset zip, you can skip cells 5–6 and unzip it directly into `data/` instead.

## Why TF 2.15?

- This repo uses `tf.keras.backend.dot` / `backend.transpose` and `tf.keras.activations.softmax(tensor)` patterns that broke in Keras 3.
- TF 2.15 is the **last TF release on Keras 2**; Keras 3 starts at TF 2.16.
- Earlier this notebook tried to pin TF 2.10 via `condacolab`, but Colab's base Python keeps moving past 3.10 and TF 2.10's wheel matrix doesn't follow. TF 2.15 ships wheels for the Python versions Colab actually serves.

## VRAM notes

| GPU | Comfortable batch size at 512×512 input |
|---|---|
| T4 (16 GB) | 16 |
| A100 (40 GB) | 32+ |
| L4 (24 GB) | 16-24 |
| RTX 4060 Ti (8 GB) | 4-8 (and even 8 OOMs in this codebase due to SSIM gradient) |

The default `batch_size: 16` in `cfs/pscn_seg.yml` works on all Colab GPUs.

## Epoch budget

The notebook runs:
- **Sanity 5 epochs** before each full run, so you catch NaN losses or OOMs in <1 hour.
- **Full 80 epochs** for both segmentation and classification.

80 + 80 ≈ 18 hours on an L4, fitting inside Colab Pro's 24 h session limit with margin. The original paper trained for 100 epochs; 80 leaves a safety buffer for the inevitable Drive-mount / preprocessing time at the start of a session. If you want closer to paper-faithful runs, push to 100 once you've seen one full run complete.

## Session persistence

Colab wipes `/content` on disconnect but Drive persists. The notebook copies the best checkpoint to Drive after each training run; section 10 shows how to restore it in a new session for evaluation.
