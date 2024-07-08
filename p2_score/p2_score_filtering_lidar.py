import os
import os.path as osp
import pickle
import sys
import copy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import ray
import torch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_args(args):
    eprint("========== filtering info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("====================================")

def filter_by_ppscore(pp_score, percentile=50, threshold=0.5):
    if len(pp_score) == 0 or np.percentile(pp_score, percentile) > threshold:
        return False
    return True

def update_pred(ptc, pp_score, preds, args):
    assert ptc.shape[0] == pp_score.shape[0]
    preds = copy.deepcopy(preds)
    preds_boxes = preds['boxes_lidar']
    box_idxs_of_pts = points_in_boxes_gpu(
        torch.from_numpy(ptc).float().cuda().unsqueeze(dim=0),
        torch.from_numpy(preds_boxes).float().cuda().unsqueeze(dim=0).contiguous()
    ).long().squeeze(dim=0).cpu()
    mask_p2 = np.array([
        filter_by_ppscore(
            pp_score[box_idxs_of_pts == i],
            percentile=args.pp_score_percentile,
            threshold=args.pp_score_threshold)
         for i in range(preds_boxes.shape[0])
    ], dtype=bool)
    if args.confidence_score_threshold > 0:
        mask_score = np.array([score > args.confidence_score_threshold for score in preds['score']])
        mask_p2_soft = np.array([
            filter_by_ppscore(
                pp_score[box_idxs_of_pts == i],
                percentile=args.pp_score_percentile,
                threshold=args.soft_pp_score_threshold)
            for i in range(preds_boxes.shape[0])
        ])
        mask_score = np.logical_and(mask_score, mask_p2_soft)
        mask_score_soft = np.array([score > args.soft_confidence_score_threshold for score in preds['score']])
        mask_p2 = np.logical_and(mask_p2, mask_score_soft)
        if args.and_operation:
            mask = np.logical_and(mask_p2, mask_score)
        else:
            mask = np.logical_or(mask_p2, mask_score)
    else:
        mask = mask_p2
    for k in preds:
        if not k in ["frame_id", "metadata"]:
            preds[k] = preds[k][mask]
    return preds

def process_one_scene(idx, args, det_bboxes):
    det_bbox = det_bboxes[idx]
    if args.dataset == "lyft":
        frame_id = int(det_bbox['frame_id'])
        pp_score_path = osp.join(args.data_paths.p2score_path, f"{frame_id:06d}.npy")
        lidar_path = osp.join(args.data_paths.scan_path, f"{frame_id:06d}.bin")
    elif args.dataset == "ithaca365":
        pp_score_path = osp.join(args.data_paths.p2score_path, f"{det_bbox['metadata']['token']}.npy")
        lidar_path = osp.join(args.data_paths.scan_path, str(det_bbox['frame_id']))
    ptc = load_velo_scan(lidar_path)[:, :3]
    pp_score = np.load(pp_score_path)
    return update_pred(
        ptc, pp_score, det_bbox,
        args=args.det_filtering
        )

@ray.remote(num_gpus=0.1)
def process_batch_scene(idx_list, args, det_bboxes):
    return [
        process_one_scene(idx, args, det_bboxes)
        for idx in idx_list
    ]


@hydra.main(config_path="configs/", config_name="p2_score_filtering.yaml")
def main(args):
    display_args(args)
    ray.init(num_cpus=args.n_processes, num_gpus=1)
    det_bboxes = pickle.load(open(args.result_path, "rb"))
    det_bboxes_new = []
    det_bboxes_id = ray.put(det_bboxes)

    idx_list = np.arange(len(det_bboxes), dtype=int)
    for idx_sublist in np.array_split(idx_list, args.n_processes):
        det_bboxes_new.append(process_batch_scene.remote(
            idx_sublist, args, det_bboxes_id))

    det_bboxes_new = ray.get(det_bboxes_new)
    det_bboxes_new = sum(det_bboxes_new, [])
    count_before = 0
    for det_bbox in det_bboxes:
        count_before += det_bbox['boxes_lidar'].shape[0]
    count_after = 0
    for det_bbox in det_bboxes_new:
        count_after += det_bbox['boxes_lidar'].shape[0]
    print(f"#bbox before: {count_before}, #bbox after: {count_after}")
    pickle.dump(det_bboxes_new, open(args.save_path, "wb"))


if __name__ == "__main__":
    main()
