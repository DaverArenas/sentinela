# Model Registry and Provenance

Model artifacts live in this directory on each edge node and are **never committed to git**.
Each model used by Sentinela must have a provenance record below, and its SHA-256 must be pinned
in `config/vision_config.yaml`; the runtime refuses to start if the file on disk does not match.

> A permissive code license does not by itself establish that pretrained weights or their training
> data are commercially unrestricted. No model is commercially approved until the review below is complete.

## Active model

### `yolox-s-coco` — YOLOX-S (COCO-80)

| Field | Value |
|---|---|
| Model name / version | YOLOX-S, release `0.1.1rc0` |
| Format | ONNX, input `images` 1×3×640×640 float32 BGR, output `output` 1×8400×85 |
| File | `models/yolox_s.onnx` |
| SHA-256 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| Original repository | https://github.com/Megvii-BaseDetection/YOLOX |
| Artifact source | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx |
| Code license | Apache-2.0 |
| Weight license | Distributed with the Apache-2.0 repository release — **to be confirmed in legal review** |
| Training data | COCO 2017 (images under individual Flickr licenses; annotations CC BY 4.0) — **restrictions to be reviewed** |
| Required attribution | Megvii YOLOX authors (Ge et al., 2021, "YOLOX: Exceeding YOLO Series in 2021") |
| Modifications | None (used as released) |
| Adapter | `vision/inference/backends/yolox.py` (family `yolox`) |
| Commercial status | **Evaluation only — review pending** |

Download (one time, requires network):

```bash
curl -L -o models/yolox_s.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
sha256sum models/yolox_s.onnx   # must match the SHA-256 above
```

## Candidates under evaluation

| Model | Status |
|---|---|
| RT-DETR (lyuwenyu/RT-DETR, Apache-2.0 code) | Not yet evaluated; needs ONNX export and a `rt-detr` adapter |

## Changing models

1. Add a provenance record above.
2. Update `model:` in `config/vision_config.yaml` (id, version, path, sha256, source, license).
3. Add an adapter under `vision/inference/backends/` if the model family is new.
4. Re-run the recorded-video tests and the RTX 3060 benchmark (`uv run sentinela-benchmark --device cuda`),
   and record the results in `applications/sentinela_vision/README.md`.
