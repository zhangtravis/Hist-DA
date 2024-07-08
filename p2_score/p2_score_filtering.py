import os
import os.path as osp
import pickle
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from utils import kitti_util
from utils.pointcloud_utils import is_within_fov, load_velo_scan, objs2label

from types import SimpleNamespace


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_args(args):
    eprint("========== filtering info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("====================================")

def predicts2objs(preds):
    objs = []
    for i in range(preds['location'].shape[0]):
        obj = SimpleNamespace()
        obj.t = preds['location'][i]
        obj.l = preds['dimensions'][i][0]
        obj.h = preds['dimensions'][i][1]
        obj.w = preds['dimensions'][i][2]
        obj.ry = preds['rotation_y'][i]
        obj.score = preds['score'][i]
        obj.obj_type = preds['name'][i]
        objs.append(obj)
    return objs

def filter_by_ppscore(ptc_rect, pp_score, obj, percentile=50, threshold=0.5):
    ry = obj.ry
    l = obj.l
    w = obj.w
    xz_center = obj.t[[0, 2]]
    ptc_xz = ptc_rect[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    y_mask = (ptc_rect[:, 1] > obj.t[1] - obj.h) * (ptc_rect[:, 1] <= obj.t[1])
    mask = mask * y_mask
    if mask.sum() == 0 or np.percentile(pp_score[mask], percentile) > threshold:
        return False
    return True

@hydra.main(config_path="configs/", config_name="p2_score_filtering.yaml")
def main(args):
    det_bboxes = pickle.load(open(args.result_path, "rb"))
    idx_list = [int(det_bbox['frame_id']) for det_bbox in det_bboxes]
    os.makedirs(args.save_path, exist_ok=True)
    count_before = 0
    count_after = 0
    for idx, det_bbox in zip(tqdm(idx_list), det_bboxes):
        assert idx == int(det_bbox['frame_id'])
        calib = kitti_util.Calibration(
            osp.join(args.data_paths.calib_path, f"{idx:06d}.txt"))
        ptc = load_velo_scan(osp.join(
            args.data_paths.scan_path, f"{idx:06d}.bin"))
        ptc_in_rect = calib.project_velo_to_rect(ptc[:, :3])
        pp_score = np.load(
            osp.join(args.data_paths.p2score_path, f"{idx:06d}.npy"))

        det_obj = predicts2objs(det_bbox)
        count_before += len(det_obj)
        det_obj = list(filter(
            lambda obj: filter_by_ppscore(
                ptc_in_rect, pp_score, obj,
                percentile=args.det_filtering.pp_score_percentile,
                threshold=args.det_filtering.pp_score_threshold),
            det_obj))
        count_after += len(det_obj)
        with open(osp.join(args.save_path, f"{idx:06d}.txt"), "w") as f:
            f.write(objs2label(det_obj, calib, with_score=True))
    print(f"#objs before: {count_before} #objs after: {count_after}")


if __name__ == "__main__":
    main()
