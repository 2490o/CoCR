# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Main script for language modulation (with offline PLY/OBJ export)."""

import os
import json
import datetime

import numpy as np
import torch
import torch.distributed as dist

local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Default to 0 if not set

from main_utils import parse_option, BaseTrainTester
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator
from models import BeaUTyDETR
from models import APCalculator

from tqdm import tqdm

import ipdb

st = ipdb.set_trace


# ============================ Offline export helpers ============================

def _boxes_to_corners_aligned_xyzwhl(boxes: torch.Tensor):
    """
    将轴对齐6D盒子 [cx,cy,cz, dx,dy,dz] 转为 8 个角点 (N,8,3).
    若你是7D带yaw，请改为先在局部坐标生成角点，再绕相应轴旋转，再平移到中心。
    """
    if boxes is None or boxes.numel() == 0:
        return None
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes)

    # 允许 1D 情况，比如 (6,)
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    c = boxes[:, 0:3]
    s = boxes[:, 3:6] / 2.0
    offs = torch.tensor([
        [-1, -1, -1], [-1, -1, 1],
        [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1],
        [1, 1, -1], [1, 1, 1],
    ], dtype=boxes.dtype, device=boxes.device) * s.unsqueeze(1)
    return c.unsqueeze(1) + offs  # (N,8,3)


def _merge_box_meshes_ply(pred_corners=None, gt_corners=None, det_corners=None):
    """
    将多个盒子的8角点合并为一个带颜色的三角网格（用于 PLY）。
    不同来源：pred=绿，gt=红，det=蓝。

    返回:
        V: (Nv, 3) float32
        C: (Nv, 3) float32  [0,1]
        F: (Nf, 3) int32
    """
    all_sets = []

    def add_boxes(corners, rgb):
        if corners is None:
            return
        if torch.is_tensor(corners):
            if corners.numel() == 0:
                return
            corners = corners.detach().cpu().numpy()
        # corners: (K, 8, 3)
        faces_template = np.array([
            [0, 1, 3], [0, 3, 2],  # bottom
            [4, 5, 7], [4, 7, 6],  # top
            [0, 1, 5], [0, 5, 4],  # side
            [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4],
            [1, 3, 7], [1, 7, 5],
        ], dtype=np.int32)
        verts_list, faces_list, cols_list = [], [], []
        v_base = 0
        rgb = np.array(rgb, dtype=np.float32)
        for k in range(corners.shape[0]):
            v = corners[k]  # (8,3)
            f = faces_template + v_base
            c = np.tile(rgb[None, :], (8, 1))
            verts_list.append(v)
            faces_list.append(f)
            cols_list.append(c)
            v_base += 8
        if verts_list:
            V = np.concatenate(verts_list, axis=0)
            F = np.concatenate(faces_list, axis=0)
            C = np.concatenate(cols_list, axis=0)
            all_sets.append((V, C, F))

    add_boxes(pred_corners, (0.0, 1.0, 0.0))  # 绿：预测
    add_boxes(gt_corners, (1.0, 0.0, 0.0))  # 红：GT
    add_boxes(det_corners, (0.0, 0.0, 1.0))  # 蓝：提议/输入框

    if not all_sets:
        return None, None, None

    # 合并
    V_all, C_all, F_all = [], [], []
    v_offset = 0
    for V, C, F in all_sets:
        V_all.append(V)
        C_all.append(C)
        F_all.append(F + v_offset)
        v_offset += V.shape[0]
    V_all = np.concatenate(V_all, axis=0)
    C_all = np.concatenate(C_all, axis=0)
    F_all = np.concatenate(F_all, axis=0)
    return V_all, C_all, F_all


def _write_ply_pointcloud(path, xyz: np.ndarray, rgb: np.ndarray = None):
    """
    保存点云到 PLY。支持 per-vertex color（[0,1] 或 [0,255]）。
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            f"Missing Open3D: pip install open3d\nOriginal error: {e}"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if rgb is not None:
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))
    o3d.io.write_point_cloud(path, pcd)


def _write_ply_mesh(path, V, C, F):
    """
    保存三角网格到 PLY（带 vertex color）。
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            f"Missing Open3D: pip install open3d\nOriginal error: {e}"
        )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    # Vertex color
    mesh.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float32))
    o3d.io.write_triangle_mesh(path, mesh)


def _write_obj_mesh(path, V, F):
    """
    简洁OBJ写法（单一材质，无 per-vertex color）。
    如需三色分别输出，建议拆成 pred/gt/det 三个 OBJ。
    """
    with open(path, 'w') as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in F:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


# ============================ Core trainer/tester ============================

class TrainTester(BaseTrainTester):

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                butd=args.butd,
                augment_det=args.augment_det
            )

        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_test else 'test',
            use_color=args.use_color,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            butd=args.butd,
        )
        return train_dataset, test_dataset

    # Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            data_path=args.data_root,
            self_attend=args.self_attend,
            use_cocr=getattr(args, 'use_cocr', False),
            cocr_cfg=dict(
                vocab_dir=args.cocr_vocab_dir, num_steps=args.cocr_steps,
                num_iters=args.cocr_iters, prune_theta=args.cocr_theta,
                use_icp=not args.cocr_no_icp, use_icg=not args.cocr_no_icg,
                use_cvfa=not args.cocr_no_cvfa, use_prune=not args.cocr_no_prune,
                detach_aux=getattr(args, 'cocr_detach', False),
            ) if getattr(args, 'use_cocr', False) else None,
            use_qhead=getattr(args, 'use_qhead', False),
        )
        return model

    # input data.
    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),  # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],  # list[B]  text
            'point_instance_label': batch_data['point_instance_label'],  # ([B, 50000])  instance label
            "det_boxes": batch_data['all_detected_boxes'],  # ([B, 132, 6]) groupfree detection boxes
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],  # ([B, 132]) mask
            "det_class_ids": batch_data['all_detected_class_ids']  # ([B, 132])  box id
        }

    # only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.
        这里集成：离线导出点云 + 检测框(PRED/GT/DET)为 PLY/OBJ
        """
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        prefixes = ['last_']

        # 导出相关可选项（无需在 parse_option 里改，这里用 getattr 提供默认）
        vis_n = int(getattr(args, "vis_n", 0))  # 0 表示不导出
        vis_out = str(getattr(args, "vis_out", "vis"))
        vis_thr = float(getattr(args, "vis_score_thresh", 0.30))
        vis_format = str(getattr(args, "vis_format", "ply")).lower()
        os.makedirs(vis_out, exist_ok=True)

        evaluator = GroundingEvaluator(
            only_root=False, thresholds=[0.25, 0.5],  # TODO only_root=True
            prefixes=prefixes
        )

        ap_calculator_list = [
            APCalculator(iou_thresh)
            for iou_thresh in args.ap_iou_thresholds  # [0.25, 0.50]
        ]

        # CoCR: evaluate several fusion weights in one pass; the first is the main one
        # score configs (CoCR fuse weight, IoU-quality weight); the first is the main one (ckpt selection)
        fuse_weights = list(args.cocr_fuse_weights) if getattr(args, 'use_cocr', False) else [0.0]
        qiou_weights = list(args.qhead_weights) if getattr(args, 'use_qhead', False) else [0.0]
        main_fw, main_qw = fuse_weights[0], qiou_weights[0]
        cfgs = [(fw, main_qw) for fw in fuse_weights[1:]] + [(main_fw, qw) for qw in qiou_weights[1:]]
        extra_eval = {
            cfg: (GroundingEvaluator(only_root=False, thresholds=[0.25, 0.5], prefixes=prefixes),
                  [APCalculator(t) for t in args.ap_iou_thresholds])
            for cfg in cfgs
        }

        # NOTE Main eval branch
        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # forward + compute loss
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )

            if evaluator is not None:
                for prefix in prefixes:
                    if prefix != 'last_':
                        continue

                    # 拿本进程（未all_gather）的预测/gt
                    batch_pred_box, batch_pred_conf, batch_gt, batch_gt_mask = \
                        evaluator.evaluate(end_points, prefix, fuse_weight=main_fw, qiou_weight=main_qw)
                    for (fw, qw), (ev_fw, ap_list_fw) in extra_eval.items():
                        pb, pc, gb, gm = ev_fw.evaluate(end_points, prefix, fuse_weight=fw, qiou_weight=qw)
                        pb, pc, gb, gm = gather_tensor(pb), gather_tensor(pc), gather_tensor(gb), gather_tensor(gm)
                        if dist.get_rank() == 0:
                            for apc in ap_list_fw:
                                apc.step(pb, pc, gb, gm)
                    vis_n = int(getattr(args, "vis_n", 0))
                    # ----------------- 离线导出（仅 rank==0；最多导出 vis_n 个样本） -----------------
                    if dist.get_rank() == 0 and vis_n > 0 and batch_idx < vis_n:
                        try:
                            bi = 0  # 本 batch 中第0个样本；你也可遍历 range(B)
                            # 点云
                            pc = end_points['point_clouds'][bi].detach().cpu().numpy()  # [N,6] or [N,3]
                            xyz = pc[:, :3]
                            rgb = pc[:, 3:6] if pc.shape[1] >= 6 else None

                            base = os.path.join(vis_out, f"{batch_idx:06d}")
                            _write_ply_pointcloud(base + "_pc.ply", xyz, rgb)

                            # 预测框
                            pred_b = batch_pred_box[bi] if batch_pred_box is not None else None
                            conf_b = batch_pred_conf[bi] if batch_pred_conf is not None else None
                            if pred_b is not None:
                                if not torch.is_tensor(pred_b):
                                    pred_b = torch.as_tensor(pred_b)
                                if conf_b is not None:
                                    if not torch.is_tensor(conf_b):
                                        conf_b = torch.as_tensor(conf_b)
                                    keep = conf_b > vis_thr
                                    if keep.ndim > 1:
                                        keep = keep.squeeze(-1)
                                    pred_b = pred_b[keep]

                            # GT 框（按 mask 过滤）
                            gt_b = batch_gt[bi] if batch_gt is not None else None
                            if gt_b is not None:
                                if not torch.is_tensor(gt_b):
                                    gt_b = torch.as_tensor(gt_b)
                                if batch_gt_mask is not None:
                                    m = batch_gt_mask[bi]
                                    if not torch.is_tensor(m):
                                        m = torch.as_tensor(m)
                                    gt_b = gt_b[m.bool()]

                            # 提议/输入框（可选，取前20个避免太密）
                            det_b = batch_data.get('all_detected_boxes', None)
                            if det_b is not None:
                                det_b = det_b[bi]
                                if not torch.is_tensor(det_b):
                                    det_b = torch.as_tensor(det_b)
                                if det_b.shape[0] > 20:
                                    det_b = det_b[:20]

                            # 角点
                            pred_c = _boxes_to_corners_aligned_xyzwhl(
                                pred_b) if pred_b is not None and pred_b.numel() > 0 else None
                            gt_c = _boxes_to_corners_aligned_xyzwhl(
                                gt_b) if gt_b is not None and gt_b.numel() > 0 else None
                            det_c = _boxes_to_corners_aligned_xyzwhl(
                                det_b) if det_b is not None and det_b.numel() > 0 else None

                            V, C, F = _merge_box_meshes_ply(pred_c, gt_c, det_c)
                            if V is not None:
                                if vis_format == "ply":
                                    _write_ply_mesh(base + "_boxes.ply", V, C, F)
                                else:
                                    _write_obj_mesh(base + "_boxes.obj", V, F)

                            # 元信息（可选）
                            meta = {
                                "num_points": int(xyz.shape[0]),
                                "pred_thresh": float(vis_thr),
                                "counts": {
                                    "pred": int(0 if pred_c is None else pred_c.shape[0]),
                                    "gt": int(0 if gt_c is None else gt_c.shape[0]),
                                    "det": int(0 if det_c is None else det_c.shape[0]),
                                }
                            }
                            with open(base + "_meta.json", "w") as f:
                                json.dump(meta, f, indent=2)

                        except Exception as e:
                            print(f"[WARN] Export failed at batch {batch_idx}: {e}")

                    # ----------------- AP 统计需要 all_gather 的版本 -----------------
                    batch_pred_box_allgpu = gather_tensor(batch_pred_box)
                    batch_pred_conf_allgpu = gather_tensor(batch_pred_conf)
                    batch_gt_allgpu = gather_tensor(batch_gt)
                    batch_gt_mask_allgpu = gather_tensor(batch_gt_mask)

                    if dist.get_rank() == 0:
                        for ap_calculator in ap_calculator_list:
                            ap_calculator.step(batch_pred_box_allgpu, batch_pred_conf_allgpu,
                                               batch_gt_allgpu, batch_gt_mask_allgpu)

        evaluator.synchronize_between_processes()
        for (fw, qw), (ev_fw, ap_list_fw) in extra_eval.items():
            ev_fw.synchronize_between_processes()
            if dist.get_rank() == 0:
                a25 = ev_fw.dets[("last_", 0.25, 'top1_acc')] / max(ev_fw.gts[("last_", 0.25, 'top1_acc')], 1)
                a50 = ev_fw.dets[("last_", 0.5, 'top1_acc')] / max(ev_fw.gts[("last_", 0.5, 'top1_acc')], 1)
                aps = [apc.compute_metrics() for apc in ap_list_fw]
                self.logger.info(
                    f'[score fuse={fw} qiou={qw}] Top1-Acc@0.25 {a25:.4f} Top1-Acc@0.5 {a50:.4f} '
                    f'AP@0.25 {aps[0]:.4f} AP@0.5 {aps[1]:.4f}'
                )
        APs = []
        if dist.get_rank() == 0:
            # evaluate Top1-Acc
            c_25 = evaluator.dets[("last_", 0.25, 'top1_acc')] / max(evaluator.gts[("last_", 0.25, 'top1_acc')], 1)
            c_50 = evaluator.dets[("last_", 0.5, 'top1_acc')] / max(evaluator.gts[("last_", 0.5, 'top1_acc')], 1)

            if not (args.eval_val or args.eval_test):
                self.tensorboard.item["val_score"]["top1_acc_0.25"] = c_25
                self.tensorboard.item["val_score"]["top1_acc_0.5"] = c_50
            print('=====================>', f'{prefixes[0]} Top1-Acc: 0.25: {c_25}, 0.5: {c_50}',
                  '<=====================')
            self.logger.info(f'[main fuse={main_fw} qiou={main_qw}] Top1-Acc@0.25 {c_25:.4f} Top1-Acc@0.5 {c_50:.4f}')

            if not (args.eval_val or args.eval_test):
                # tensorboard
                for key in self.tensorboard.item["val_loss"]:
                    self.tensorboard.item["val_loss"][key] = stat_dict[key] / len(test_loader)
                self.tensorboard.dump_tensorboard("val_loss", epoch)

            for i, ap_calculator in enumerate(ap_calculator_list):
                ap = ap_calculator.compute_metrics()
                self.logger.info(
                    '=====================>'
                    f'{prefixes[0]} IOU THRESH: {args.ap_iou_thresholds[i]}, AP: {ap}'
                    '<====================='
                )
                APs.append(ap)

            if not (args.eval_val or args.eval_test):
                # tensorboard
                for i in range(len(args.ap_iou_thresholds)):
                    self.tensorboard.item["val_score"][f"AP_{args.ap_iou_thresholds[i]}"] = APs[i]
                self.tensorboard.dump_tensorboard("val_score", epoch)

        return APs


# ============================ DDP utils ============================

def is_dist_avail_and_initialized():
    """
    Returns:
        True if distributed training is enabled
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns:
        The number of processes in the process group
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def gather_tensor(data):
    if is_dist_avail_and_initialized():
        tensor_list = [torch.empty_like(data) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, data)
        gathered_tensor = torch.cat(tensor_list, dim=0)
    else:
        gathered_tensor = data
    return gathered_tensor


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# ============================ Main ============================

if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    opt = parse_option()

    # 下面四个可选项，如果你没在 parse_option 里声明，也OK（有默认值）
    # 建议：在命令行里加上它们，例如：
    #   --vis_n 5 --vis_out vis --vis_score_thresh 0.3 --vis_format ply
    if not hasattr(opt, "vis_n"): setattr(opt, "vis_n", 0)  # 0 = 不导出
    if not hasattr(opt, "vis_out"): setattr(opt, "vis_out", "vis")
    if not hasattr(opt, "vis_score_thresh"): setattr(opt, "vis_score_thresh", 0.30)
    if not hasattr(opt, "vis_format"): setattr(opt, "vis_format", "ply")

    # distributed
    # torch.cuda.set_device(opt.local_rank)
    torch.cuda.set_device(local_rank)
    setup_for_distributed(local_rank == 0)

    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        timeout=datetime.timedelta(seconds=int(os.environ.get('NCCL_TIMEOUT_S', 5400)))
    )

    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    _ = train_tester.main(opt)
