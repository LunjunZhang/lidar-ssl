import time

import numba
import numpy as np

import torch
import torch.nn.functional as tf


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel_plus(
        points,
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        point_idx_to_voxel_idx,
        max_points=35,
        max_voxels=20000,
    ):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)

    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
        # now take notes on where this point is put
        point_idx_to_voxel_idx[i] = voxelidx
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel_plus(
        points,
        voxel_size,
        coors_range,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxels,
        coors,
        point_idx_to_voxel_idx,
        max_points=35,
        max_voxels=20000, 
    ):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
        point_idx_to_voxel_idx[i] = voxelidx
    return voxel_num


def points_to_voxel_plus(
        points,
        voxel_size,
        coors_range,
        max_points=35,
        reverse_index=True,
        max_voxels=20000,
    ):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    point_idx_to_voxel_idx = -np.ones((points.shape[0], ))
    
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel_plus(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, point_idx_to_voxel_idx, 
            max_points, max_voxels)
    else:
        voxel_num = _points_to_voxel_kernel_plus(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, point_idx_to_voxel_idx,
            max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel, point_idx_to_voxel_idx


@numba.jit(nopython=True)
def bound_points_jit_flow_plus(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices


# deal with a single instance at first 

def single_sample_flow_3d_to_bev(
        points, 
        scene_flow, 
        voxel_size,
        coors_range,
        max_points=35,
        reverse_index=True,
        max_voxels=20000,
    ):
    # logic here:
    # voxelize frame_1, and then voxelize warped frame_1, 
    # then voxelize warped frame_1, voxelize
    # calculate the difference in warped positions
    # 
    # voxels: [M, max_points, ndim] float tensor. only contain points.
    # coors: [M, 3] int32 tensor.
    # num_points_per_voxel: [M] int32 tensor.

    assert points.shape[1] == scene_flow.shape[1] == 3
    assert points.shape[0] == scene_flow.shape[0]
    warped_points = points + scene_flow
    
    voxels_1, coors_1, num_points_per_voxel_1, point_idx_to_voxel_idx_1 = points_to_voxel_plus(
        points,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    voxels_2, coors_2, num_points_per_voxel_2, point_idx_to_voxel_idx_2 = points_to_voxel_plus(
        warped_points,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    
    # M = voxels_1.shape[0]
    M = voxels_2.shape[0]
    N = points.shape[0]

    # we should calculate the inverse warping
    # which means the following corrsponds the warped frame (which looks like the second frame)

    flow_voxel = np.zeros_like(voxels_2)
    flow_voxel_count = np.zeros((M, ), dtype=np.int32)
    flow_voxel_vec_sum = np.zeros((M, 3))

    for i in range(N):
        voxel_idx_1 = point_idx_to_voxel_idx_1[i]
        voxel_idx_2 = point_idx_to_voxel_idx_2[i]
        if (voxel_idx_1 == -1) or (voxel_idx_2 == -1):
            continue
        voxel_idx_1 = int(voxel_idx_1)
        voxel_idx_2 = int(voxel_idx_2)
        flow_vector_i = coors_2[voxel_idx_2] - coors_1[voxel_idx_1]
        if reverse_index:
            flow_vector_i = flow_vector_i[::-1]
        # flow_vector_i should always be xyz format
        # since it's inverse warping, we take the negative sign
        if flow_voxel_count[voxel_idx_2] < max_points and flow_voxel_count[voxel_idx_2] < flow_voxel.shape[1]:
            flow_voxel[voxel_idx_2, flow_voxel_count[voxel_idx_2]] = -flow_vector_i
            flow_voxel_count[voxel_idx_2] += 1
            flow_voxel_vec_sum[voxel_idx_2, :] += -flow_vector_i
    
    flow_voxel_mean = np.zeros((M, 3))
    for i in range(M):
        if flow_voxel_count[i] > 0:
            flow_voxel_mean[i] = flow_voxel_vec_sum[i] / flow_voxel_count[i]
    
    # grid_size: a size 3 vector
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    bev_grid_flow = np.zeros((grid_size[0], grid_size[1], 2))

    for voxel_id in range(M):
        grid_x, grid_y = 0, 0
        if reverse_index:
            grid_x = coors_2[voxel_id, 2]
        else:
            grid_x = coors_2[voxel_id, 0]
        grid_y = coors_2[voxel_id, 1]
        assert 0<=grid_x<= grid_size[0] and 0<=grid_y<=grid_size[1] 
        '''
        warped_grid_x = grid_x + flow_voxel_mean[voxel_id, 0]
        warped_grid_y = grid_y + flow_voxel_mean[voxel_id, 1]
        bev_grid_flow[grid_x, grid_y, 0] = warped_grid_x
        bev_grid_flow[grid_x, grid_y, 1] = warped_grid_y
        '''
        bev_grid_flow[grid_x, grid_y, 0] = flow_voxel_mean[voxel_id, 0]
        bev_grid_flow[grid_x, grid_y, 1] = flow_voxel_mean[voxel_id, 1]
    
    return bev_grid_flow


def single_sample_flow_3d_to_bev_v2(
        points, 
        scene_flow, 
        voxel_size,
        coors_range,
        max_points=35,
        reverse_index=True,
        max_voxels=20000,
        points_frame2=None,
    ):
    assert points.shape[1] == scene_flow.shape[1] == 3
    assert points.shape[0] == scene_flow.shape[0]
    warped_points = points + scene_flow
    
    voxels_1, coors_1, num_points_per_voxel_1, point_idx_to_voxel_idx_1 = points_to_voxel_plus(
        points,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    voxels_2, coors_2, num_points_per_voxel_2, point_idx_to_voxel_idx_2 = points_to_voxel_plus(
        warped_points,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    f2_voxels, f2_coors, f2_num_points_per_voxel, f2_point_idx_to_voxel_idx = points_to_voxel_plus(
        points_frame2,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    
    M = voxels_2.shape[0]
    N = points.shape[0]

    # we should calculate the inverse warping
    # which means the following corrsponds the warped frame (which looks like the second frame)

    flow_voxel_count = np.zeros((M, ), dtype=np.int32)
    flow_voxel_vec_sum = np.zeros((M, 3))

    for i in range(N):
        voxel_idx_1 = point_idx_to_voxel_idx_1[i]
        voxel_idx_2 = point_idx_to_voxel_idx_2[i]
        if (voxel_idx_1 == -1) or (voxel_idx_2 == -1):
            continue
        voxel_idx_1 = int(voxel_idx_1)
        voxel_idx_2 = int(voxel_idx_2)
        flow_vector_i = scene_flow[i] / voxel_size
        # flow_vector_i should always be xyz format
        # since it's inverse warping, we take the negative sign
        if flow_voxel_count[voxel_idx_2] < voxels_2.shape[1]:
            flow_voxel_count[voxel_idx_2] += 1
            flow_voxel_vec_sum[voxel_idx_2, :] += -flow_vector_i
    
    flow_voxel_mean = np.zeros((M, 3))
    for i in range(M):
        if flow_voxel_count[i] > 0:
            flow_voxel_mean[i] = flow_voxel_vec_sum[i] / flow_voxel_count[i]
    
    # grid_size: a size 3 vector
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    print('coors_range', coors_range)
    print('voxel_size', voxel_size)
    print('grid_size', grid_size)

    bev_grid_pts = np.zeros((grid_size[0], grid_size[1]))
    bev_grid_flow = np.zeros((grid_size[0], grid_size[1], 2))

    for voxel_id in range(M):
        if flow_voxel_count[voxel_id] == 0:
            continue
        grid_x, grid_y = 0, 0
        if reverse_index:
            grid_x = coors_2[voxel_id, 2]
        else:
            grid_x = coors_2[voxel_id, 0]
        grid_y = coors_2[voxel_id, 1]
        assert 0<=grid_x<= grid_size[0] and 0<=grid_y<=grid_size[1] 
        bev_grid_flow[grid_x, grid_y, 0] = flow_voxel_mean[voxel_id, 0]
        bev_grid_flow[grid_x, grid_y, 1] = flow_voxel_mean[voxel_id, 1]
        bev_grid_pts[grid_x, grid_y] = 1.0
    
    bev_grid_frame1 = np.zeros((grid_size[0], grid_size[1]))
    for voxel_id in range(voxels_1.shape[0]):
        grid_x, grid_y = 0, 0
        if reverse_index:
            grid_x = coors_1[voxel_id, 2]
        else:
            grid_x = coors_1[voxel_id, 0]
        grid_y = coors_1[voxel_id, 1]
        bev_grid_frame1[grid_x, grid_y] = 1.0
    
    bev_grid_frame2 = np.zeros((grid_size[0], grid_size[1]))
    for voxel_id in range(f2_voxels.shape[0]):
        grid_x, grid_y = 0, 0
        if reverse_index:
            grid_x = f2_coors[voxel_id, 2]
        else:
            grid_x = f2_coors[voxel_id, 0]
        grid_y = f2_coors[voxel_id, 1]
        bev_grid_frame2[grid_x, grid_y] = 1.0
    
    return bev_grid_flow, dict(
        bev_grid_pts=bev_grid_pts, 
        bev_grid_frame1=bev_grid_frame1, 
        bev_grid_frame2=bev_grid_frame2,
        warped_points=points + scene_flow,
    )


def project_to_bev_2d(
        pts, flow, 
        voxel_size, coors_range, 
        max_points,
        reverse_index,
        max_voxels,
    ):
    voxels, coors, num_points_per_voxel, point_idx_to_voxel_idx = points_to_voxel_plus(
        pts,
        voxel_size,
        coors_range,
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    assert pts.shape[0] == flow.shape[0]
    assert pts.shape[1] == flow.shape[1] == 3
    M = voxels.shape[0]
    flow_voxel_count = np.zeros((M, ), dtype=np.int32)
    flow_voxel_vec_sum = np.zeros((M, 3))
    flow_voxel_mean = np.zeros((M, 3))
    N = pts.shape[0]
    for i in range(N):
        voxel_idx = point_idx_to_voxel_idx[i]
        if (voxel_idx == -1 or voxel_idx >= M):
            continue
        voxel_idx = int(voxel_idx)
        flow_vector_i = flow[i] / voxel_size
        flow_voxel_count[voxel_idx] += 1
        flow_voxel_vec_sum[voxel_idx, :] += flow_vector_i
    for i in range(M):
        if flow_voxel_count[i] > 0:
            flow_voxel_mean[i] = flow_voxel_vec_sum[i] / flow_voxel_count[i]
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    bev_pts = np.zeros((grid_size[0], grid_size[1]))
    bev_flow = np.zeros((grid_size[0], grid_size[1], 2))
    for voxel_id in range(M):
        if flow_voxel_count[voxel_id] == 0:
            continue
        grid_x = coors[voxel_id, 2] if reverse_index else coors[voxel_id, 0]
        grid_y = coors[voxel_id, 1]
        assert 0<=grid_x<= grid_size[0] and 0<=grid_y<=grid_size[1] 
        bev_flow[grid_x, grid_y, 0] = flow_voxel_mean[voxel_id, 0]
        bev_flow[grid_x, grid_y, 1] = flow_voxel_mean[voxel_id, 1]
        bev_pts[grid_x, grid_y] = 1.0
    bev_pts = torch.from_numpy(bev_pts).rot90(1, [0, 1]).cpu().numpy()
    bev_flow = torch.from_numpy(bev_flow).rot90(1, [0, 1]).cpu().numpy()
    return bev_pts, bev_flow


def warp_images(img, flow):
    H, W = img.shape[2:]
    grid = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device))
    grid = torch.stack(grid[::-1], dim=0).unsqueeze(0).expand(img.shape[0], -1, -1, -1)
    grid = grid + flow
    grid[:, 0] = grid[:, 0] * 2.0 / (W - 1) - 1.0
    grid[:, 1] = grid[:, 1] * 2.0 / (H - 1) - 1.0
    return tf.grid_sample(img.float(), grid.permute(0, 2, 3, 1).float(), align_corners=True)


def flow_consistency(flow1, flow2):
    H, W = flow1.shape[2:]
    grid = torch.meshgrid(torch.arange(H, device=flow1.device), torch.arange(W, device=flow1.device))
    grid = torch.stack(grid[::-1], dim=0).unsqueeze(0).expand(flow1.shape[0], -1, -1, -1)
    warped_grid = warp_images(warp_images(grid.float(), flow1), flow2)
    mask = ((grid - warped_grid).norm(dim=1) <= 5).unsqueeze(1)
    return mask


def warp_images_np(img, flow):
    assert len(img.shape) == 2
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
    return warp_images(img, flow).squeeze().cpu().numpy()


def flow_consistency_np(flow1, flow2):
    flow1 = torch.from_numpy(flow1).permute(2, 0, 1).unsqueeze(0)
    flow2 = torch.from_numpy(flow2).permute(2, 0, 1).unsqueeze(0)
    return flow_consistency(flow1, flow2).squeeze().cpu().numpy()


def single_sample_flow_to_bev_v3(
        pts_1, 
        pts_2, 
        flow_1to2, 
        flow_2to1, 
        voxel_size,
        coors_range,
        max_points=35,
        reverse_index=True,
        max_voxels=20000,
    ):
    assert pts_1.shape[1] == flow_1to2.shape[1] == 3
    assert pts_1.shape[0] == flow_1to2.shape[0]
    assert pts_2.shape[1] == flow_2to1.shape[1] == 3
    assert pts_2.shape[0] == flow_2to1.shape[0]

    bev_img_1, bev_flow_1to2 = project_to_bev_2d(
        pts=pts_1, flow=flow_1to2, 
        voxel_size=voxel_size, coors_range=coors_range, 
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    warped_1to2 = warp_images_np(bev_img_1, bev_flow_1to2)
    bev_img_2, bev_flow_2to1 = project_to_bev_2d(
        pts=pts_2, flow=flow_2to1, 
        voxel_size=voxel_size, coors_range=coors_range, 
        max_points=max_points,
        reverse_index=reverse_index,
        max_voxels=max_voxels,
    )
    warped_2to1 = warp_images_np(bev_img_2, bev_flow_2to1)
    consistency_mask = flow_consistency_np(bev_flow_1to2, bev_flow_2to1)

    threshold_1to2 = warped_1to2.copy()
    threshold_1to2[threshold_1to2 > 0.4] = 1.0
    threshold_1to2[threshold_1to2 < 0.9] = 0.0
    
    threshold_2to1 = warped_2to1.copy()
    threshold_2to1[threshold_2to1 > 0.4] = 1.0
    threshold_2to1[threshold_2to1 < 0.9] = 0.0
    
    return dict(
        bev_img_1=bev_img_1, 
        bev_flow_1to2=bev_flow_1to2, 
        bev_img_2=bev_img_2,
        bev_flow_2to1=bev_flow_2to1,
        warped_1to2=warped_1to2,
        warped_2to1=warped_2to1,
        consistency_mask=consistency_mask,
        warp_diff_1=np.power(bev_img_1 - threshold_2to1, 2),
        warp_diff_2=np.power(bev_img_2 - threshold_1to2, 2),
        bev_img_diff=np.power(bev_img_1 - bev_img_2, 2),
        warped_3d_1to2=pts_1 + flow_1to2,
        warped_3d_2to1=pts_2 + flow_2to1,
        overlap_1=0.7 * bev_img_1 + 0.3 * threshold_2to1, 
        overlap_2=0.7 * bev_img_2 + 0.3 * threshold_1to2,
        threshold_1to2=threshold_1to2,
        threshold_2to1=threshold_2to1,
    )
