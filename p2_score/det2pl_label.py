import os
import os.path as osp
import pickle
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from utils import kitti_util
from utils.pointcloud_utils import objs2label

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

@hydra.main(config_path="configs/", config_name="p2_score_filtering.yaml")
def main(args):
    det_bboxes = pickle.load(open(args.result_path, "rb"))
    idx_list = [int(det_bbox['frame_id']) for det_bbox in det_bboxes]
    os.makedirs(args.save_path, exist_ok=True)
    for idx, det_bbox in zip(tqdm(idx_list), det_bboxes):
        assert idx == int(det_bbox['frame_id'])
        calib = kitti_util.Calibration(
            osp.join(args.data_paths.calib_path, f"{idx:06d}.txt"))
        det_obj = predicts2objs(det_bbox)
        with open(osp.join(args.save_path, f"{idx:06d}.txt"), "w") as f:
            f.write(objs2label(det_obj, calib, with_score=True))


if __name__ == "__main__":
    main()
