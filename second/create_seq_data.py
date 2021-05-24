import copy
import pathlib
import pickle
import os

import fire
import numpy as np
from skimage import io as imgio

from second.core import box_np_ops
from second.core.point_cloud.point_cloud_ops import bound_points_jit

# use the seq_kitti_common
from second.data import seq_kitti_common as kitti

from second.utils.progress_bar import list_bar as prog_bar
"""
Note: tqdm has problem in my system(win10), so use my progress bar
try:
    from tqdm import tqdm as prog_bar
except ImportError:
    from second.utils.progress_bar import progress_bar_iter as prog_bar
"""


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(data_path, infos, relative_path, remove_outside=True, num_features=4):
    for seq_id, info in infos.items():
        rect = info['calib/R0_rect']
        Trv2c = info['calib/Tr_velo_to_cam']
        P2 = info['calib/P2']
        
        for frame_id, frame_info in info['frames'].items():
            if 'frame_idx' not in frame_info:
                print('create_data.py _calculate_num_points_in_gt', 'seq_id=', seq_id, 'frame_id=', frame_id)
                print('frame_idx not in frame_info keys')
                continue
                # import pdb; pdb.set_trace()
            elif frame_id != frame_info['frame_idx']:
                print('create_data.py _calculate_num_points_in_gt', 'seq_id=', seq_id, 'frame_id=', frame_id)
                import pdb; pdb.set_trace()
            if relative_path:
                v_path = str(pathlib.Path(data_path) / frame_info["velodyne_path"])
            else:
                v_path = frame_info["velodyne_path"]
            points_v = np.fromfile(
                v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
            if remove_outside:
                points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, frame_info["img_shape"])
            # points_v = points_v[points_v[:, 0] > 0]
            if 'annos' not in frame_info:
                import pdb; pdb.set_trace()
            annos = frame_info['annos']
            num_obj = len([n for n in annos['name'] if n != 'DontCare'])
            # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
            dims = annos['dimensions'][:num_obj]
            loc = annos['location'][:num_obj]
            rots = annos['rotation_y'][:num_obj]
            gt_boxes_camera = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
                gt_boxes_camera, rect, Trv2c)
            indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
            num_points_in_gt = indices.sum(0)
            num_ignored = len(annos['dimensions']) - num_obj
            num_points_in_gt = np.concatenate(
                [num_points_in_gt, -np.ones([num_ignored])])
            annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(data_path,
                           save_path=None,
                           create_trainval=False,
                           relative_path=True):
    train_seq_ids = _read_imageset_file("./second/data/ImageSets/train.txt")
    val_seq_ids = _read_imageset_file("./second/data/ImageSets/val.txt")
    trainval_seq_ids = _read_imageset_file("./second/data/ImageSets/trainval.txt")
    test_seq_ids = _read_imageset_file("./second/data/ImageSets/test.txt")

    seqlen_train = _read_imageset_file("./second/data/ImageSets/seqlen_train.txt")
    seqlen_val = _read_imageset_file("./second/data/ImageSets/seqlen_val.txt")
    seqlen_trainval = _read_imageset_file("./second/data/ImageSets/seqlen_trainval.txt")
    seqlen_test = _read_imageset_file("./second/data/ImageSets/seqlen_test.txt")

    print("Generate info. this may take several minutes.")
    
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    # import pdb; pdb.set_trace()
    
    kitti_infos_train = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        seq_ids=train_seq_ids,
        seq_lengths=seqlen_train,
        #image_ids=train_img_ids,
        relative_path=relative_path)
    # kitti_infos_train is not a list anymore, it is a dict()
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        #image_ids=val_img_ids,
        seq_ids=val_seq_ids,
        seq_lengths=seqlen_val,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    kitti_infos_trainval = kitti_infos_train.copy()
    kitti_infos_trainval.update(kitti_infos_val.copy())
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_trainval, f)
    
    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        #image_ids=test_img_ids,
        seq_ids=test_seq_ids,
        seq_lengths=seqlen_test,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)


def _create_reduced_point_cloud(data_path, info_path, save_path=None, back=False):
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos.values()):
        this_seq_id = info['seq_idx']
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        for frame_id, frame_info in info['frames'].items():
            print('Creating reduced PC: seq_id=', this_seq_id, 'frame_id=', frame_id, 'info keys=', frame_info.keys())
            if ('annos' not in frame_info.keys()) or ('frame_idx' not in frame_info.keys()) or ('velodyne_path' not in frame_info.keys()):
                print('NOT ENOUGH KEYS ...')
                continue
            v_path = frame_info['velodyne_path']
            v_path = pathlib.Path(data_path) / v_path
            points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
            # first remove z < 0 points
            # keep = points_v[:, -1] > 0
            # points_v = points_v[keep]
            # then remove outside.
            if back:
                points_v[:, 0] = -points_v[:, 0]
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2, frame_info["img_shape"])
            # v_path : pathlib.Path('training') / info_type / seq_idx_str / img_idx_str
            # PREVIOUSLY : pathlib.Path('training') / info_type / img_idx_str
            if save_path is None:
                # save_filename = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
                path_training = v_path.parent.parent.parent
                path_info_type = v_path.parent.parent.stem
                path_seq_id = v_path.parent.stem
                save_file_dir = path_training / (str(path_info_type) + "_reduced") / path_seq_id
                os.makedirs(save_file_dir, exist_ok=True)
                save_filename = save_file_dir / v_path.name
                # save_filename = str(v_path) + '_reduced'
                print(path_training, str(path_info_type), path_seq_id, save_filename)
                # import pdb; pdb.set_trace()
                if back:
                    save_filename += "_back"
            else:
                save_filename = str(pathlib.Path(save_path) / v_path.name)
                if back:
                    save_filename += "_back"
            print('Saving to path:', save_filename)
            with open(save_filename, 'w') as f:
                points_v.tofile(f)


def create_reduced_point_cloud(
        data_path,
        train_info_path=None,
        val_info_path=None,
        test_info_path=None,
        save_path=None,
        with_back=False):
    if train_info_path is None:
        train_info_path = pathlib.Path(data_path) / 'kitti_infos_train.pkl'
    if val_info_path is None:
        val_info_path = pathlib.Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = pathlib.Path(data_path) / 'kitti_infos_test.pkl'

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


def create_groundtruth_database(
        data_path,
        info_path=None,
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None,
        relative_path=True,
        lidar_only=False,
        bev_only=False,
        coors_range=None):
    root_path = pathlib.Path(data_path)
    if info_path is None:
        info_path = root_path / 'kitti_infos_train.pkl'
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = pathlib.Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "kitti_dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    all_db_infos = {}
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = []
    group_counter = 0
    for info in prog_bar(kitti_infos.values()):
        this_seq_id = info['seq_idx']
        num_features = 4
        if 'pointcloud_num_features' in info:
            num_features = info['pointcloud_num_features']
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        for frame_id, frame_info in info['frames'].items():
            print('Creating GT databse: seq_id=', this_seq_id, 'frame_id=', frame_id, 'info keys=', frame_info.keys())
            if ('annos' not in frame_info.keys()) or ('frame_idx' not in frame_info.keys()) or ('velodyne_path' not in frame_info.keys()):
                print('NOT ENOUGH KEYS ...')
                continue
            velodyne_path = frame_info['velodyne_path']
            if relative_path:
                # velodyne_path = str(root_path / velodyne_path) + "_reduced"
                velodyne_path = str(root_path / velodyne_path)
            points = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape([-1, num_features])
            frame_idx = frame_info["frame_idx"]
            assert frame_idx == frame_id
            if not lidar_only:
                points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2, frame_info["img_shape"])
            annos = frame_info["annos"]
            names = annos["name"]
            bboxes = annos["bbox"]
            difficulty = annos["difficulty"]
            gt_idxes = annos["index"]
            num_obj = np.sum(annos["index"] >= 0)
            rbbox_cam = kitti.anno_to_rbboxes(annos)[:num_obj]
            rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
            if bev_only: # set z and h to limits
                assert coors_range is not None
                rbbox_lidar[:, 2] = coors_range[2]
                rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]
            group_dict = {}
            group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
            if "group_ids" in annos:
                group_ids = annos["group_ids"]
            else:
                group_ids = np.arange(bboxes.shape[0], dtype=np.int64)
            point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)
            for i in range(num_obj):
                filename = f"{this_seq_id}_{frame_id}_{names[i]}_{gt_idxes[i]}.bin"
                filepath = database_save_path / filename
                print('Saving object to:', filepath)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= rbbox_lidar[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
                if names[i] in used_classes:
                    if relative_path:
                        db_path = str(database_save_path.stem + "/" + filename)
                    else:
                        db_path = str(filepath)
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        # "image_idx": image_idx,
                        "seq_idx": this_seq_id, "frame_idx": frame_id,
                        "gt_idx": gt_idxes[i],
                        "box3d_lidar": rbbox_lidar[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                        # "group_id": -1,
                        # "bbox": bboxes[i],
                    }
                    local_group_id = group_ids[i]
                    # if local_group_id >= 0:
                    if local_group_id not in group_dict:
                        group_dict[local_group_id] = group_counter
                        group_counter += 1
                    db_info["group_id"] = group_dict[local_group_id]
                    if "score" in annos:
                        db_info["score"] = annos["score"][i]
                    all_db_infos[names[i]].append(db_info)
    
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")
    
    print('Saving database info to:', db_info_save_path)
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


if __name__ == '__main__':
    fire.Fire()
