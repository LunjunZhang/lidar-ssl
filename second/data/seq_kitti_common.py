import concurrent.futures as futures
import os
import pathlib
import re
from collections import OrderedDict

import numpy as np
from skimage import io


def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union


def get_seq_index_str(seq_idx):
    return "{:04d}".format(seq_idx)


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_kitti_info_path(
        seq_idx,
        img_idx,
        prefix,
        info_type='image_02',
        file_tail='.png',
        training=True,
        relative_path=True,
        exist_check=True):
    seq_idx_str = get_seq_index_str(seq_idx)
    img_idx_str = get_image_index_str(img_idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / seq_idx_str / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / seq_idx_str / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        #import pdb; pdb.set_trace()
        print('get_kitti_info_path ValueError. file not exist: {}'.format(prefix/file_path))
        raise ValueError("file not exist: {}".format(prefix/file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_kitti_info_path_v2(
        seq_idx,
        prefix,
        info_type='label_02',
        file_tail='.txt',
        training=True,
        relative_path=True,
        exist_check=True):
    seq_idx_str = get_seq_index_str(seq_idx)
    seq_idx_str += file_tail

    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / seq_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / seq_idx_str
    if exist_check and not (prefix / file_path).exists():
        # import pdb; pdb.set_trace()
        print('get_kitti_info_path_v2: ValueError. file not exist: {}'.format(prefix/file_path))
        raise ValueError("file not exist: {}".format(prefix/file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(seq_idx, frame_idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(
        seq_idx, frame_idx, prefix, 'image_02', '.png', training,
        relative_path, exist_check)


# each sequence only has one file
def get_label_path(seq_idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path_v2(
        seq_idx, prefix, 'label_02', '.txt', training,
        relative_path, exist_check)


def get_velodyne_path(seq_idx, frame_idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(
        seq_idx, frame_idx, prefix, 'velodyne', '.bin', training,
        relative_path, exist_check)


# each sequence only has one file
def get_calib_path(seq_idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path_v2(
        seq_idx, prefix, 'calib', '.txt', training,
        relative_path, exist_check)


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(
        path,
        training=True,
        label_info=True,
        velodyne=False,
        calib=False,
        seq_ids=15,
        seq_lengths=None,
        #image_ids=7481,
        extend_matrix=True,
        num_worker=8,
        relative_path=True,
        with_imageshape=True):
    # image_infos = []
    root_path = pathlib.Path(path)
    if not isinstance(seq_ids, list):
        seq_ids = list(range(seq_ids))
    assert (seq_lengths is not None) and (len(seq_lengths) == len(seq_ids))
    
    all_seq_infos = dict()
    '''
        all_seq_infos: 
        seq_id -> seq_info (dict)
            seq_info: 
                'seq_idx' : this_seq_id
                'pointcloud_num_features' : 4
                'calib/P0' : P0
                'calib/P1' : P1
                'calib/P2' : P2
                'calib/P3' : P3
                'calib/R0_rect' : rect_4x4
                'calib/Tr_velo_to_cam' : Tr_velo_to_cam
                'calib/Tr_imu_to_velo' : Tr_imu_to_velo
                'frames' : dict()
                    frame_id (int) : seq_info[frame_id] (dict)
                        'annos' : frame_annos (dict)
                        'frame_idx' : frame_id
                        'velodyne_path' : get_velodyne_path
                        'img_path' : get_image_path
                        'img_shape' : np.array(io.imread(img_path).shape[:2], dtype=np.int32)
    '''
    for this_seq_id in seq_ids:
        print('this_seq_id=', this_seq_id)
    
    for seq_list_idx, this_seq_id in enumerate(seq_ids):
        print('this_seq_id=', this_seq_id)
        seq_info = dict()
        seq_info['seq_idx'] = this_seq_id
        seq_info['pointcloud_num_features'] = 4
        seq_info['frames'] = dict()

        seq_annotations = None  # just for one single sequence
        if label_info:
            label_path = get_label_path(
                seq_idx=this_seq_id, prefix=path, training=training, relative_path=relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            seq_annotations = get_label_anno(label_path)
            # seq_annotations: dict (frame_id -> annotation for each frame)
            # seq_lengths[this_seq_id] = len(seq_annotations)
            '''if len(seq_annotations) != seq_lengths[this_seq_id] + 1:
                import pdb; pdb.set_trace()'''
            for frame_id, frame_annos in seq_annotations.items():
                if frame_id not in seq_info['frames']:
                    seq_info['frames'][frame_id] = dict()
                seq_info['frames'][frame_id]['annos'] = frame_annos
                add_difficulty_to_annos(seq_info['frames'][frame_id])
        
        if calib:
            calib_path = get_calib_path(
                seq_idx=this_seq_id, prefix=path, training=training, relative_path=relative_path)
            if relative_path:
                calib_path = str(root_path / calib_path)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            seq_info['calib/P0'] = P0
            seq_info['calib/P1'] = P1
            seq_info['calib/P2'] = P2
            seq_info['calib/P3'] = P3
            R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            seq_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            seq_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            seq_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        
        for frame_id in range(seq_lengths[seq_list_idx] + 1):
            if frame_id not in seq_info['frames']:
                continue
            if 'annos' not in seq_info['frames'][frame_id]:
                continue
            frame_info = dict()
            frame_info['frame_idx'] = frame_id
            if velodyne:
                try:
                    frame_info['velodyne_path'] = get_velodyne_path(
                        seq_idx=this_seq_id, frame_idx=frame_id, prefix=path, 
                        training=training, relative_path=relative_path, exist_check=True
                    )
                except ValueError:
                    print('Trouble with seq_id=', this_seq_id, 'frame_id=', frame_id, 'velodyne')
                    #import pdb; pdb.set_trace()
                    continue
            frame_info['img_path'] = get_image_path(
                seq_idx=this_seq_id, frame_idx=frame_id, prefix=path, 
                training=training, relative_path=relative_path, exist_check=True
            )
            if with_imageshape:
                img_path = frame_info['img_path']
                if relative_path:
                    img_path = str(root_path / img_path)
                frame_info['img_shape'] = np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32)
            if frame_id not in seq_info['frames']:
                seq_info['frames'][frame_id] = dict()
            seq_info['frames'][frame_id].update(frame_info)
            print('Finished seq_id=', this_seq_id, 'frame_id=', frame_id, 'seq_length', seq_lengths)
        
        all_seq_infos[this_seq_id] = seq_info
    return all_seq_infos


def label_str_to_int(labels, remove_dontcare=True, dtype=np.int32):
    class_to_label = get_class_to_label_map()
    ret = np.array([class_to_label[l] for l in labels], dtype=dtype)
    if remove_dontcare:
        ret = ret[ret > 0]
    return ret


def get_class_to_label_map():
    class_to_label = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 3,
        'Person_sitting': 4,
        'Truck': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,
    }
    return class_to_label


def get_classes():
    return get_class_to_label_map().keys()


def filter_gt_boxes(gt_boxes, gt_labels, used_classes):
    mask = np.array([l in used_classes for l in gt_labels], dtype=np.bool)
    return mask


def filter_anno_by_mask(image_anno, mask):
    img_filtered_annotations = {}
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][mask])
    return img_filtered_annotations


def filter_infos_by_used_classes(infos, used_classes):
    new_infos = []
    for info in infos:
        annos = info["annos"]
        name_in_info = False
        for name in used_classes:
            if name in annos["name"]:
                name_in_info = True
                break
        if name_in_info:
            new_infos.append(info)
    return new_infos


def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations


def remove_low_height(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['bbox']) if (s[3] - s[1]) >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations


def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['score']) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations


def keep_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds


def drop_arrays_by_name(gt_names, used_classes):
    inds = [
        i for i, x in enumerate(gt_names) if x not in used_classes
    ]
    inds = np.array(inds, dtype=np.int64)
    return inds


def apply_mask_(array_dict):
    pass


def filter_kitti_anno(image_anno,
                      used_classes,
                      used_difficulty=None,
                      dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple, np.ndarray)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations


def filter_annos_class(image_annos, used_class):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['name']) if x in used_class
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_difficulty(image_annos, used_difficulty):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno['difficulty']) if x in used_difficulty
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_height(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['bbox']) if (s[3] - s[1]) >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_empty_annos(image_annos):
    new_image_annos = []
    for anno in image_annos:
        if anno["name"].shape[0] > 0:
            new_image_annos.append(anno.copy())
    return new_image_annos


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def annos_to_kitti_label(annos):
    num_instance = len(annos["name"])
    result_lines = []
    for i in range(num_instance):
        result_dict = {
            'name': annos["name"][i],
            'truncated': annos["truncated"][i],
            'occluded': annos["occluded"][i],
            'alpha':annos["alpha"][i],
            'bbox': annos["bbox"][i],
            'dimensions': annos["dimensions"][i],
            'location': annos["location"][i],
            'rotation_y': annos["rotation_y"][i],
        }
        line = kitti_result_line(result_dict)
        result_lines.append(line)
    return result_lines


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def add_difficulty_to_annos_v2(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = not ((occlusion > max_occlusion[0]) or (height < min_height[0])
                 or (truncation > max_trunc[0]))
    moderate_mask = not ((occlusion > max_occlusion[1]) or (height < min_height[1])
                 or (truncation > max_trunc[1]))
    hard_mask = not ((occlusion > max_occlusion[2]) or (height < min_height[2])
                 or (truncation > max_trunc[2]))
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path):
    all_frame_annotations = dict()

    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    all_frames_content = [line.strip().split(' ') for line in lines]
    content_dict = dict()
    for content_line in all_frames_content:
        frame_id = int(content_line[0])
        if frame_id not in content_dict:
            content_dict[frame_id] = list()
        content_dict[frame_id].append(content_line.copy())
    
    offset = 2
    
    for frame_id, content in content_dict.items():
        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': []
        })
        num_objects = len([x[0+offset] for x in content if x[0+offset] != 'DontCare'])
        annotations['name'] = np.array([x[0+offset] for x in content])
        num_gt = len(annotations['name'])
        annotations['truncated'] = np.array([float(x[1+offset]) for x in content])
        annotations['occluded'] = np.array([int(x[2+offset]) for x in content])
        annotations['alpha'] = np.array([float(x[3+offset]) for x in content])
        annotations['bbox'] = np.array(
            [[float(info) for info in x[4+offset:8+offset]] for x in content]).reshape(-1, 4)
        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations['dimensions'] = np.array(
            [[float(info) for info in x[8+offset:11+offset]] for x in content]).reshape(
                -1, 3)[:, [2, 0, 1]]
        annotations['location'] = np.array(
            [[float(info) for info in x[11+offset:14+offset]] for x in content]).reshape(-1, 3)
        annotations['rotation_y'] = np.array(
            [float(x[14+offset]) for x in content]).reshape(-1)
        if len(content) != 0 and len(content[0]) == 16+offset:  # have score
            annotations['score'] = np.array([float(x[15+offset]) for x in content])
        else:
            annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
        all_frame_annotations[frame_id] = annotations
    
    return all_frame_annotations


def get_pseudo_label_anno():
    annotations = {}
    annotations.update({
        'name': np.array(['Car']),
        'truncated': np.array([0.0]),
        'occluded': np.array([0]),
        'alpha': np.array([0.0]),
        'bbox': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'dimensions': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'location': np.array([[0.1, 0.1, 15.0]]),
        'rotation_y': np.array([[0.1, 0.1, 15.0]])
    })
    return annotations


def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations


def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + '.txt')
        anno = get_label_anno(label_filename)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos


def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
