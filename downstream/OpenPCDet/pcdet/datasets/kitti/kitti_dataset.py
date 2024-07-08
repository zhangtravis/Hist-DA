import copy
import pickle

import numpy as np

from torch.utils.data import dataset
from skimage import io
import os
import os.path as osp
import MinkowskiEngine as ME
import torch

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]


class KittiDataset(DatasetTemplate):
    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / (
            "training" if self.split != "test" else "testing"
        )

        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        self.sample_id_list = (
            [x.strip() for x in open(split_dir).readlines()]
            if split_dir.exists()
            else None
        )

        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        self.load_p2_score = None
        self.load_history_path = None
        self.load_background_sample = None
        self.contant_reflex = dataset_cfg.get("CONSTANT_REFLEX", False)
        if "load_p2_score" in dataset_cfg:
            self.load_p2_score = dataset_cfg.load_p2_score
            self.load_p2_test_score = dataset_cfg.load_p2_test_score
        if "load_background_sample" in dataset_cfg:
            self.load_background_sample = dataset_cfg.load_background_sample
        if "LOAD_HISTORY" in dataset_cfg:
            self.load_history = copy.deepcopy(dataset_cfg.LOAD_HISTORY)
            self.history_cache_dir = None
            if "CACHE_ROOT" in self.load_history:
                if self.load_history.get("HISTORY_AUG", False):
                    self.history_cache_dir = osp.join(
                        self.load_history.CACHE_ROOT,
                        f"raw_points_fwonly={self.load_history.FORWARD_ONLY}"
                        f"_history_scans_path={osp.basename(osp.normpath(self.load_history.DATA_PATH))}",
                    )
                else:
                    self.history_cache_dir = osp.join(
                        self.load_history.CACHE_ROOT,
                        f"fwonly={self.load_history.FORWARD_ONLY}_vs={self.load_history.VOXEL_SIZE:02f}"
                        f"_history_scans_path={osp.basename(osp.normpath(self.load_history.DATA_PATH))}",
                    )
                if not os.path.exists(self.history_cache_dir):
                    os.makedirs(self.history_cache_dir, exist_ok=True)
                    os.chmod(self.history_cache_dir, 0o777)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info("Loading KITTI dataset")
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info("Total samples for KITTI dataset: %d" % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split
        self.root_split_path = self.root_path / (
            "training" if self.split != "test" else "testing"
        )

        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        self.sample_id_list = (
            [x.strip() for x in open(split_dir).readlines()]
            if split_dir.exists()
            else None
        )
        # print(split_dir)
        # print(self.sample_id_list)
        # raise

    def get_history_raw(self, idx):
        assert self.load_history is not None
        history_scans = None
        if self.history_cache_dir is not None and osp.exists(
            osp.join(self.history_cache_dir, f"{idx}.pth")
        ):
            try:
                history_scans = torch.load(
                    osp.join(self.history_cache_dir, f"{idx}.pth")
                )
            except:
                print("reading error " + idx)
        if history_scans is None:
            history_scans = pickle.load(
                open(osp.join(self.load_history.DATA_PATH, f"{idx}.pkl"), "rb")
            )
            history_scans = list(history_scans.values())
            trans_mat = np.load(
                osp.join(self.load_history.TRANS_MAT_PATH, f"{idx}.npy")
            )
            history_scans = [
                transform_points(x, np.linalg.inv(trans_mat)) for x in history_scans
            ]
            if self.load_history.FORWARD_ONLY:
                history_scans = [x[x[:, 0] > 0, :] for x in history_scans]
            if self.history_cache_dir is not None:
                torch.save(
                    history_scans, osp.join(self.history_cache_dir, f"{idx}.pth")
                )
                os.chmod(osp.join(self.history_cache_dir, f"{idx}.pth"), 0o777)

        if self.training and self.load_history.get("RANDOM_DROPOUT", False):
            num_scans = np.random.randint(1, high=len(history_scans) + 1)
            _scan_num_choice = np.random.choice(
                len(history_scans), num_scans, replace=False
            )
            history_scans = [history_scans[i] for i in _scan_num_choice]

        if (
            self.load_history.LIMIT_NUM > 0
            and len(history_scans) > self.load_history.LIMIT_NUM
        ):
            if self.training:
                _scan_num_choice = np.random.choice(
                    len(history_scans), self.load_history.LIMIT_NUM, replace=False
                )
            else:
                _scan_num_choice = range(self.load_history.LIMIT_NUM)
            history_scans = [history_scans[i] for i in _scan_num_choice]

        return history_scans

    def get_history(self, idx):
        assert self.load_history is not None
        if self.history_cache_dir is not None and osp.exists(
            osp.join(self.history_cache_dir, f"{idx}.pth")
        ):
            history_coordinates = torch.load(
                osp.join(self.history_cache_dir, f"{idx}.pth")
            )
        else:
            history_scans = pickle.load(
                open(osp.join(self.load_history.DATA_PATH, f"{idx}.pkl"), "rb")
            )
            history_scans = list(history_scans.values())
            trans_mat = np.load(
                osp.join(self.load_history.TRANS_MAT_PATH, f"{idx}.npy")
            )
            history_scans = [
                transform_points(x, np.linalg.inv(trans_mat)) for x in history_scans
            ]
            if self.load_history.FORWARD_ONLY:
                history_scans = [x[x[:, 0] > 0, :] for x in history_scans]
            history_coordinates = [
                ME.utils.sparse_quantize(x / self.load_history.VOXEL_SIZE)
                for x in history_scans
            ]
            if self.history_cache_dir is not None:
                torch.save(
                    history_coordinates, osp.join(self.history_cache_dir, f"{idx}.pth")
                )
                os.chmod(osp.join(self.history_cache_dir, f"{idx}.pth"), 0o777)

        if (
            self.load_history.LIMIT_NUM > 0
            and len(history_coordinates) > self.load_history.LIMIT_NUM
        ):
            if self.training:
                _scan_num_choice = np.random.choice(
                    len(history_coordinates), self.load_history.LIMIT_NUM, replace=False
                )
            else:
                _scan_num_choice = range(self.load_history.LIMIT_NUM)
            history_coordinates = [history_coordinates[i] for i in _scan_num_choice]

        history_features = [torch.ones((len(x), 1)) for x in history_coordinates]
        return history_coordinates, history_features

    def get_background_sample(self, idx):
        return np.load(osp.join(self.load_background_sample, f"{idx}.npy"))

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / "velodyne" / ("%s.bin" % idx)
        assert lidar_file.exists(), lidar_file
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        if self.contant_reflex:
            points[:, 3] = self.contant_reflex
        # if self.load_ephemerality is not None:
        #     # print(f"{idx} loading ephemerality")
        #     if self.training:
        #         load_ephemerality = self.load_ephemerality.train
        #     else:
        #         load_ephemerality = self.load_ephemerality.val
        #     pts_intensity = np.load(os.path.join(
        #         load_ephemerality, f"{idx}.npy")).astype(np.float32)
        #     pts_intensity[pts_intensity < 0] = 0
        #     # points[:, 3] = pts_intensity
        #     if len(pts_intensity.shape) == 1:
        #         pts_intensity = pts_intensity.reshape(-1,1)
        #     points = np.concatenate((points[:, :3], pts_intensity), axis=1)
        return points

    def get_p2_score(self, idx):
        assert self.load_p2_score is not None
        if self.training:
            return np.load(os.path.join(self.load_p2_score, f"{idx}.npy")).astype(
                np.float32
            )
        else:
            return np.load(os.path.join(self.load_p2_test_score, f"{idx}.npy")).astype(
                np.float32
            )

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / "image_2" / ("%s.png" % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / "image_2" / ("%s.png" % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / "label_2" / ("%s.txt" % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / "depth_2" / ("%s.png" % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / "calib" / ("%s.txt" % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / "planes" / ("%s.txt" % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, "r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]

        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(
        self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None
    ):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            # print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {"num_features": 4, "lidar_idx": sample_idx}
            info["point_cloud"] = pc_info

            image_info = {
                "image_idx": sample_idx,
                "image_shape": self.get_image_shape(sample_idx),
            }
            info["image"] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.0
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate(
                [calib.V2C, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
            )
            calib_info = {"P2": P2, "R0_rect": R0_4x4, "Tr_velo_to_cam": V2C_4x4}

            info["calib"] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations["name"] = np.array([obj.cls_type for obj in obj_list])
                annotations["truncated"] = np.array(
                    [obj.truncation for obj in obj_list]
                )
                annotations["occluded"] = np.array([obj.occlusion for obj in obj_list])
                annotations["alpha"] = np.array([obj.alpha for obj in obj_list])
                annotations["bbox"] = (
                    np.concatenate(
                        [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0
                    )
                    if len(obj_list) > 0
                    else np.array([]).reshape(0, 4)
                )
                annotations["dimensions"] = np.array(
                    [[obj.l, obj.h, obj.w] for obj in obj_list]
                ).reshape(
                    -1, 3
                )  # lhw(camera) format
                annotations["location"] = (
                    np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                    if len(obj_list) > 0
                    else np.array([]).reshape(0, 3)
                )
                annotations["rotation_y"] = np.array([obj.ry for obj in obj_list])
                annotations["score"] = np.array([obj.score for obj in obj_list])
                annotations["difficulty"] = np.array(
                    [obj.level for obj in obj_list], np.int32
                )

                num_objects = len(
                    [obj.cls_type for obj in obj_list if obj.cls_type != "DontCare"]
                )
                num_gt = len(annotations["name"])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations["index"] = np.array(index, dtype=np.int32)

                if len(obj_list) > 0:
                    loc = annotations["location"][:num_objects]
                    dims = annotations["dimensions"][:num_objects]
                    rots = annotations["rotation_y"][:num_objects]
                    loc_lidar = calib.rect_to_lidar(loc)
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    loc_lidar[:, 2] += h[:, 0] / 2
                    gt_boxes_lidar = np.concatenate(
                        [loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])],
                        axis=1,
                    )
                    annotations["gt_boxes_lidar"] = gt_boxes_lidar
                else:
                    annotations["gt_boxes_lidar"] = np.array([]).reshape(0, 7)

                info["annos"] = annotations

                if count_inside_pts:
                    if len(obj_list) > 0:
                        points = self.get_lidar(sample_idx)
                        calib = self.get_calib(sample_idx)
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])

                        fov_flag = self.get_fov_flag(
                            pts_rect, info["image"]["image_shape"], calib
                        )
                        pts_fov = points[fov_flag]
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                        for k in range(num_objects):
                            flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations["num_points_in_gt"] = num_points_in_gt
                    else:
                        annotations["num_points_in_gt"] = np.ones(0, dtype=np.int32)

            return info

        sample_id_list = (
            sample_id_list if sample_id_list is not None else self.sample_id_list
        )
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(
        self, info_path=None, used_classes=None, split="train"
    ):
        import torch

        database_save_path = Path(self.root_path) / (
            "gt_database" if split == "train" else ("gt_database_%s" % split)
        )
        db_info_save_path = Path(self.root_path) / ("kitti_dbinfos_%s.pkl" % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info["point_cloud"]["lidar_idx"]
            points = self.get_lidar(sample_idx)
            annos = info["annos"]
            names = annos["name"]
            difficulty = annos["difficulty"]
            bbox = annos["bbox"]
            gt_boxes = annos["gt_boxes_lidar"]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = "%s_%s_%d.bin" % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                # print(filepath, gt_points.shape)
                with open(filepath, "w") as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(
                        filepath.relative_to(self.root_path)
                    )  # gt_database/xxxxx.bin
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                        "bbox": bbox[i],
                        "score": annos["score"][i],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)

    def update_groundtruth_database(
        self,
        source_info_path=None,
        det_info_path=None,
        used_classes=None,
        split="train",
    ):
        import torch

        database_save_path = Path(self.root_path) / (
            "gt_database" if split == "train" else ("gt_database_%s" % split)
        )
        db_info_save_path = Path(self.root_path) / ("kitti_dbinfos_%s.pkl" % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(source_info_path, "rb") as f:
            infos = pickle.load(f)

        with open(det_info_path, "rb") as f:
            det_infos = pickle.load(f)

        assert len(det_infos) == len(infos)
        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info["point_cloud"]["lidar_idx"]
            points = self.get_lidar(sample_idx)
            # annos = info['annos']
            annos = det_infos[k]
            assert annos["frame_id"] == sample_idx
            del annos["frame_id"]
            annos["gt_boxes_lidar"] = annos["boxes_lidar"]
            del annos["boxes_lidar"]

            gt_boxes = annos["gt_boxes_lidar"]
            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
            num_points_in_gt = np.zeros(num_obj, dtype=np.int32)
            for i in range(num_obj):
                num_points_in_gt[i] = (point_indices[i] > 0).sum()
            annos["num_points_in_gt"] = num_points_in_gt
            annos["difficulty"] = np.zeros(num_obj, dtype=np.int32) - 1
            annos["index"] = np.arange(num_obj, dtype=np.int32)
            annos["score"] = np.zeros(num_obj, dtype=np.float32) - 1

            names = annos["name"]
            difficulty = annos["difficulty"]
            bbox = annos["bbox"]

            info["annos"] = annos

            for i in range(num_obj):
                filename = "%s_%s_%d.bin" % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                # print(filepath, gt_points.shape)
                with open(filepath, "w") as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(
                        filepath.relative_to(self.root_path)
                    )  # gt_database/xxxxx.bin
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                        "bbox": bbox[i],
                        "score": annos["score"][i],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)

        with open(Path(self.root_path) / osp.basename(source_info_path), "wb") as f:
            pickle.dump(infos, f)

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "truncated": np.zeros(num_samples),
                "occluded": np.zeros(num_samples),
                "alpha": np.zeros(num_samples),
                "bbox": np.zeros([num_samples, 4]),
                "dimensions": np.zeros([num_samples, 3]),
                "location": np.zeros([num_samples, 3]),
                "rotation_y": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros([num_samples, 7]),
                "cls_pred": np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict["calib"][batch_index]
            image_shape = batch_dict["image_shape"][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, calib
            )
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["alpha"] = (
                -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0])
                + pred_boxes_camera[:, 6]
            )
            pred_dict["bbox"] = pred_boxes_img.reshape(-1, 4)
            pred_dict["dimensions"] = pred_boxes_camera[:, 3:6].reshape(-1, 3)
            pred_dict["location"] = pred_boxes_camera[:, 0:3].reshape(-1, 3)
            pred_dict["rotation_y"] = pred_boxes_camera[:, 6]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes

            if "roi_cls_preds" in box_dict:
                pred_dict["cls_pred"] = box_dict["roi_cls_preds"].cpu().numpy()

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict["frame_id"][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict["frame_id"] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ("%s.txt" % frame_id)
                with open(cur_det_file, "w") as f:
                    bbox = single_pred_dict["bbox"]
                    loc = single_pred_dict["location"]
                    dims = single_pred_dict["dimensions"]  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            "%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                            % (
                                single_pred_dict["name"][idx],
                                single_pred_dict["alpha"][idx],
                                bbox[idx][0],
                                bbox[idx][1],
                                bbox[idx][2],
                                bbox[idx][3],
                                dims[idx][1],
                                dims[idx][2],
                                dims[idx][0],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                single_pred_dict["rotation_y"][idx],
                                single_pred_dict["score"][idx],
                            ),
                            file=f,
                        )

        return annos

    def evaluation(
        self, det_annos, class_names, range_eval=True, ranges=(0, 30, 50, 80), **kwargs
    ):
        if "annos" not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.kitti_infos]
        # np.concatenate([a["bbox"] for a in eval_det_annos], 0)
        # print(id(eval_det_annos), "evaluation")
        if range_eval:
            ap_result_str, ap_dict = kitti_eval.get_range_eval_result(
                eval_gt_annos, eval_det_annos, class_names, ranges=ranges
            )
        else:
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                eval_gt_annos, eval_det_annos, class_names
            )

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info["point_cloud"]["lidar_idx"]
        img_shape = info["image"]["image_shape"]
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get("GET_ITEM_LIST", ["points"])

        input_dict = {
            "frame_id": sample_idx,
            "calib": calib,
        }

        if "annos" in info:
            annos = info["annos"]
            annos = common_utils.drop_info_with_name(annos, name="DontCare")
            loc, dims, rots = (
                annos["location"],
                annos["dimensions"],
                annos["rotation_y"],
            )
            if len(annos["name"]) > 0:
                gt_names = annos["name"]
                gt_boxes_camera = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, calib
                )

                input_dict.update({"gt_names": gt_names, "gt_boxes": gt_boxes_lidar})
            else:
                input_dict.update(
                    {
                        "gt_names": annos["name"],
                        "gt_boxes": np.zeros((0, 7), dtype=float),
                    }
                )
            if "gt_boxes2d" in get_item_list:
                input_dict["gt_boxes2d"] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict["road_plane"] = road_plane

        if "p2_score" in get_item_list:
            input_dict["p2_score"] = self.get_p2_score(sample_idx)

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
                if "p2_score" in input_dict:
                    input_dict["p2_score"] = input_dict["p2_score"][fov_flag]
            input_dict["points"] = points
            if "p2_score" in input_dict:
                assert input_dict["p2_score"].shape[0] == input_dict["points"].shape[0]

        if "images" in get_item_list:
            input_dict["images"] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict["depth_maps"] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            (
                input_dict["trans_lidar_to_cam"],
                input_dict["trans_cam_to_img"],
            ) = kitti_utils.calib_to_matricies(calib)

        if "history_scans" in get_item_list:
            if self.load_history.get("HISTORY_AUG", False):
                input_dict["history_scans"] = self.get_history_raw(sample_idx)
            else:
                (
                    input_dict["history_coordinates"],
                    input_dict["history_features"],
                ) = self.get_history(sample_idx)

        if "background_sample" in get_item_list and self.training:
            input_dict["background_sample"] = self.get_background_sample(sample_idx)

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict["image_shape"] = img_shape
        return data_dict


def create_kitti_infos(
    dataset_cfg, class_names, data_path, save_path, if_val=True, workers=4
):
    dataset = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=False,
    )
    train_split, val_split = "train", "val"

    train_filename = save_path / ("kitti_infos_%s.pkl" % train_split)
    val_filename = save_path / ("kitti_infos_%s.pkl" % val_split)
    trainval_filename = save_path / "kitti_infos_trainval.pkl"
    test_filename = save_path / "kitti_infos_test.pkl"

    print("---------------Start to generate data infos---------------")

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(
        num_workers=workers, has_label=True, count_inside_pts=True
    )
    print(f"train size: {len(kitti_infos_train)}")
    with open(train_filename, "wb") as f:
        pickle.dump(kitti_infos_train, f)
    print("Kitti info train file is saved to %s" % train_filename)
    if if_val:
        dataset.set_split(val_split)
        kitti_infos_val = dataset.get_infos(
            num_workers=workers, has_label=True, count_inside_pts=True
        )
        print(f"val size: {len(kitti_infos_val)}")
        with open(val_filename, "wb") as f:
            pickle.dump(kitti_infos_val, f)
        print("Kitti info val file is saved to %s" % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train + kitti_infos_val, f)
    # print('Kitti info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_test, f)
    # print('Kitti info test file is saved to %s' % test_filename)

    print(
        "---------------Start create groundtruth database for data augmentation---------------"
    )
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print("---------------Data preparation Done---------------")


if __name__ == "__main__":
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == "create_kitti_infos":
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        data_path = dataset_cfg.DATA_PATH if len(sys.argv) < 4 else sys.argv[3]
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=["Car", "Pedestrian", "Cyclist"],
            data_path=ROOT_DIR / "tools" / data_path,
            save_path=ROOT_DIR / "tools" / data_path,
            if_val=False if len(sys.argv) < 5 else sys.argv[4] == "True",
        )
    elif sys.argv.__len__() > 1 and sys.argv[1] == "update_groundtruth_database":
        # python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/lyft_dataset.yaml ../data/lyft_${iter_name} <source_traininfo> <detection_pseudo_labels>
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        data_path = sys.argv[3]
        dataset = KittiDataset(
            dataset_cfg=dataset_cfg,
            class_names=["Car", "Pedestrian", "Cyclist"],
            root_path=ROOT_DIR / "tools" / data_path,
            training=False,
        )
        dataset.update_groundtruth_database(
            source_info_path=sys.argv[4], det_info_path=sys.argv[5], split="train"
        )
