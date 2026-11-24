# Evaluate a CoCR-v2 checkpoint on the val or test split (single GPU).
#   DATA_ROOT=/path/to/data_root CKPT=/path/to/ckpt_best.pth SPLIT=test sh scripts/test_cocr_v2.sh
# Optional: GPU=0 PORT=29501 OUT_ID=cocr_v2_eval_test LOG_DIR=./output/
# The "IOU THRESH ... AP" lines and the [main ...] Top1 line use gamma = 0.35 (final setting);
# an extra [score fuse=0.0 qiou=0.0] line reports the scores without IoU-quality re-scoring.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Intent3D data root}
CKPT=${CKPT:?set CKPT to the checkpoint path}
SPLIT=${SPLIT:-test}
GPU=${GPU:-0}
PORT=${PORT:-29501}
OUT_ID=${OUT_ID:-cocr_v2_eval_${SPLIT}}
LOG_DIR=${LOG_DIR:-./output/}

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port ${PORT} \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${DATA_ROOT} \
    --val_freq 1 --batch_size 8 --print_freq 10 \
    --lr_backbone=1e-3 --lr=1e-4 --text_encoder_lr=1e-5 --lr_decay_epochs 60 \
    --dataset scanintend --test_dataset scanintend \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --use_cocr --cocr_steps 4 --cocr_iters 2 --cocr_theta 0.2 --cocr_detach \
    --cocr_fuse_weights 0.0 \
    --use_qhead --qhead_w 1.0 --qhead_weights 0.35 0.0 \
    --log_dir ${LOG_DIR} \
    --exp_id "${OUT_ID}" \
    --tensorboard ${LOG_DIR}/tensorboard/ \
    --pp_checkpoint ${DATA_ROOT}/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 130 --eval_batch_size 16 --num_workers 4 --num_target 50 \
    --eval_${SPLIT} --checkpoint_path ${CKPT}
