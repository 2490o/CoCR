# CoCR: Chain-of-Causal Reasoning for 3D Intent Grounding

Code and project page for **Detect What You Need: Chain-of-Causal Reasoning for 3D Intent Grounding**.

- Authors: Zihao Zhang, Aming Wu, Yang Li, Yahong Han
- OpenReview: https://openreview.net/forum?id=JF39l5WGlk
- Paper: https://openreview.net/pdf?id=JF39l5WGlk
- Project page: https://2490o.github.io/CoCR/ (`index.html` and `assets/images/` in this repository; click a figure on the page to enlarge it)

## What this code implements

The code is built on the [Intent3D](https://github.com/WeitaiKang/Intent3D) IntentNet baseline (BUTD-DETR style detector with GroupFree box proposals). CoCR is implemented in `models/cocr.py`:

- **ICP** (Intent Causal Parsing): 4 ordered reasoning-step queries decoded from the RoBERTa intent features and projected onto a K=40 functional vocabulary, giving P(f|Q).
- **ICG** (Intent Causal Grounding): function-mediated candidate scores from category-name embeddings and a train-split category-to-function prior, with dependency pruning (θ=0.2).
- **CVFA** (Causal-Visual Feature Alignment): 2 iterations of cross-attention from candidate box features to function-aware representations, graph re-estimation and dependency-weighted consistency loss.

The released configuration (`scripts/train_cocr_v2.sh`) additionally uses:

- `--cocr_detach`: the CoCR auxiliary losses (ICP / ICG / query / consistency / pruning) only train CoCR's own heads. CVFA's refined candidate features remain in the forward path and are trained by the grounding loss.
- `--use_qhead`: an IoU-quality head on every decoder layer (BCE to the IoU of each query box). At evaluation the confidence is re-scored as `conf * sigmoid(qiou)^γ`, γ = 0.35.
- CoCR's query-level function score is not fused into the ranking (`--cocr_fuse_weights 0.0`).

The functional vocabulary and targets are provided in `data/cocr/`. They are built by `data/build_cocr_vocab.py`; the vocabulary and the category-to-function prior use the training split only.

## Reference results (this release)

Single training run, 2 × RTX 3090, total batch size 16, 130 epochs, checkpoint selected on the val split (epoch 74), γ = 0.35.

| Split | Top1@0.25 | Top1@0.5 | AP@0.25 | AP@0.5 |
|---|---:|---:|---:|---:|
| Val | 58.07 | 41.18 | 45.88 | 30.27 |
| Test | 59.37 | 43.09 | 46.45 | 31.60 |

## 1. Installation

Tested with Python 3.9, PyTorch 2.1.1 (CUDA 12.1), transformers 4.37 and spaCy 3.3. `environment.yml` is the original Intent3D environment (Python 3.7 / PyTorch 1.7).

```bash
conda create -n cocr python=3.9 -y
conda activate cocr
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy transformers==4.37.0 termcolor ipdb tensorboardX h5py plyfile tabulate six tqdm
pip install spacy==3.3.3
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
sh init.sh   # compile the PointNet++ CUDA ops
```

`roberta-base` is downloaded automatically from the Hugging Face Hub on first use.

## 2. Data preparation

Data preparation is identical to Intent3D:

```
[DATA_ROOT]
├── train_v3scans.pkl          # packaged ScanNet train scans
├── val_v3scans.pkl            # packaged ScanNet val scans
├── intention_sentence/        # Intent3D annotations
├── group_free_pred_bboxes/    # GroupFree detected boxes
└── gf_detector_l6o256.pth     # PointNet++ checkpoint
```

- **ScanNet**: download ScanNet v2 (`_vh_clean_2.ply`, `_vh_clean_2.labels.ply`, `.aggregation.json`, `_vh_clean_2.0.010000.segs.json`, `.txt`). Then package it:
  ```bash
  python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
  ```
- **Intent3D annotations**: [Google Drive](https://drive.google.com/drive/folders/1qcXQXngRVmDtAOOOa_2mhZ-ZOkP_XuDp?usp=sharing).
- **GroupFree detected boxes**: [OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBjpUjJm7TkADS8vg?e=1AXJdR).
- **PointNet++ checkpoint**: [OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8).

(Optional) Rebuild the CoCR functional vocabulary and targets in `data/cocr/`:

```bash
python data/build_cocr_vocab.py --data_root [DATA_ROOT]
```

## 3. Training

Run all commands from the repository root.

```bash
DATA_ROOT=[DATA_ROOT] GPUS=0,1 sh scripts/train_cocr_v2.sh
```

- The default is 2 GPUs × batch size 8. On 4 GPUs use `NGPU=4 BS=4` to keep the total batch size at 16.
- Checkpoints (`ckpt_best.pth`, `ckpt_last.pth`) and `log.txt` are written to `./output/scanintend/cocr_v2/`.
- The validation set is evaluated after every epoch. Each `[score fuse=0.0 qiou=γ]` line reports a different γ.
- Training takes about 30 min per epoch on 2 × RTX 3090.

## 4. Evaluation

```bash
# validation split
DATA_ROOT=[DATA_ROOT] CKPT=./output/scanintend/cocr_v2/ckpt_best.pth SPLIT=val GPU=0 sh scripts/test_cocr_v2.sh
# test split
DATA_ROOT=[DATA_ROOT] CKPT=./output/scanintend/cocr_v2/ckpt_best.pth SPLIT=test GPU=0 sh scripts/test_cocr_v2.sh
```

The `IOU THRESH: 0.25 / 0.5, AP` lines and the `[main fuse=0.0 qiou=0.35]` Top-1 line give the final results. The `[score fuse=0.0 qiou=0.0]` line reports the same model without IoU-quality re-scoring.

## Acknowledgements

This codebase is built on [Intent3D](https://github.com/WeitaiKang/Intent3D), which in turn builds on [EDA](https://github.com/yanmin-wu/EDA) and [BUTD-DETR](https://github.com/nickgkan/butd_detr). We thank the authors for releasing their code.

## License

The code is released under CC BY-NC-SA 4.0 (see `LICENSE`), following the Intent3D codebase.
