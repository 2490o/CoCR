# Train CoCR-v2 on Intent3D (ScanIntend).
#   DATA_ROOT=/path/to/data_root sh scripts/train_cocr_v2.sh
# Optional: GPUS=0,1 NGPU=2 BS=8 PORT=29500 EXP_ID=cocr_v2 LOG_DIR=./output/ EXTRA="..."
# Total batch size = NGPU x BS = 16 (2 x 8 by default; use NGPU=4 BS=4 on four GPUs).
# Checkpoints and log.txt go to ${LOG_DIR}/scanintend/${EXP_ID}/ ; ckpt_best.pth is selected by
# val AP@0.5 with gamma = 0.5 (the first value of --qhead_weights), as in our run.
# Variants via EXTRA / flags:
#   qhead only (no CoCR):       remove the --use_cocr ... --cocr_fuse_weights lines
#   CoCR without stop-gradient: remove --cocr_detach
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Intent3D data root}
GPUS=${GPUS:-0,1}
NGPU=${NGPU:-2}
BS=${BS:-8}
PORT=${PORT:-29500}
EXP_ID=${EXP_ID:-cocr_v2}
LOG_DIR=${LOG_DIR:-./output/}
EXTRA=${EXTRA:-}

CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
    --nproc_per_node ${NGPU} --master_port ${PORT} \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${DATA_ROOT} \
    --val_freq 1 --batch_size ${BS} --print_freq 10 \
    --lr_backbone=1e-3 --lr=1e-4 --text_encoder_lr=1e-5 --lr_decay_epochs 60 \
    --dataset scanintend --test_dataset scanintend \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --use_cocr --cocr_steps 4 --cocr_iters 2 --cocr_theta 0.2 --cocr_detach \
    --cocr_fuse_weights 0.0 \
    --use_qhead --qhead_w 1.0 --qhead_weights 0.5 0.0 0.25 0.35 \
    ${EXTRA} \
    --log_dir ${LOG_DIR} \
    --exp_id "${EXP_ID}" \
    --tensorboard ${LOG_DIR}/tensorboard/ \
    --pp_checkpoint ${DATA_ROOT}/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 130 --eval_batch_size 16 --num_workers 8 --num_target 50
