# CoCR: Chain-of-Causal Reasoning for 3D Intent Grounding

Code and project page for **Detect What You Need: Chain-of-Causal Reasoning for 3D Intent Grounding**.

- Authors: Zihao Zhang, Aming Wu, Yang Li, Yahong Han
- Paper: https://arxiv.org/pdf/2611.23754
- Project page: https://2490o.github.io/CoCR/

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
