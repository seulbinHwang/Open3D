# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import numpy as np
import open3d as o3d
import time

from config import ConfigParser
from common import (get_default_dataset, load_rgbd_file_names, save_poses,
                    load_intrinsic, extract_trianglemesh, extract_rgbd_frames)


def slam(depth_file_names, color_file_names, intrinsic, config):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config.voxel_size, 16,
                                       config.block_count, T_frame_to_model,
                                       device)
    depth_ref = o3d.t.io.read_image(depth_file_names[0])
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                             intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                               depth_ref.columns, intrinsic,
                                               device)

    poses = []

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)
        model.update_frame_pose(i, T_frame_to_model)
        model.synthesize_model_frame(raycast_frame, config.depth_scale,
                                     config.depth_min, config.depth_max,
                                     config.trunc_voxel_multiplier, False)
        if i > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                config.depth_scale,
                                                config.depth_max,
                                                config.odometry_distance_thr)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.integrate(input_frame, config.depth_scale, config.depth_max,
                        config.trunc_voxel_multiplier)

        stop = time.time()
        print('{:04d}/{:04d} slam takes {:.4}s'.format(i, n_files,
                                                       stop - start))

    return model.voxel_grid, poses


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, bedroom, jack_jack]',
               default='lounge')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config = get_default_dataset(config)

    # Extract RGB-D frames and intrinsic from bag file.
    if config.path_dataset.endswith(".bag"):
        assert os.path.isfile(
            config.path_dataset), f"File {config.path_dataset} not found."
        print("Extracting frames from RGBD video file")
        config.path_dataset, config.path_intrinsic, config.depth_scale = extract_rgbd_frames(
            config.path_dataset)

    depth_file_names, color_file_names = load_rgbd_file_names(config)
    intrinsic = load_intrinsic(config)

    if not os.path.exists(config.path_npz):
        """ TODO
        - depth image 와 color image 에 요구되는 포맷을 알아야 함
          - depth를 rgb에 sync 맞춰서 저장하고 있음.
          - rgb: 8-bit JPG (rs.format.bgr8) -> 즉 파일에는 bgr 순서로 저장되어 있음
          - depth: 16-bit JPG ( 각 값은 mm 단위로 표현됨)
        - intrinsic 이 요구되는 포맷을 알아야 함
        
        - 확인 방법
          - ros2 bag 파일을 만드는 과정을 보면 알 수 있겠다.
          - 제공되는 dataset 을 보면 알 수 있겠다.
            - http://redwood-data.org/indoor/dataset.html
        """
        """ TODO: config
name: Default reconstruction system config
fragment_size: 100
    - 각 재구성 조각(fragment)마다 몇 개의 프레임을 사용할지 설정
    - 3D 재구성 시스템은 입력 데이터를 조각으로 나누어 처리할 수 있으며, 이 값이 클수록 큰 조각이 만들어집니다.
device: CUDA:0
engine: tensor
multiprocessing: false
    - check
path_dataset: ''
    - TODO
depth_folder: depth
color_folder: color
path_intrinsic: ''
    - TODO
path_color_intrinsic: ''
    - Check
depth_min: 0.1
    - TODO
depth_max: 3.0
    - TODO
depth_scale: 1000.0
    - TODO
odometry_method: hybrid
    - 깊이 이미지와 컬러 이미지의 정합(odometry) 방법을 설정
    - hybrid는 일반적으로 깊이와 색상 정보를 모두 사용해 정합을 수행
odometry_loop_interval: 10
    - 정합 과정에서 루프 클로징을 수행하는 간격을 설정합니다. 
    - 예를 들어, 이 값을 10으로 설정하면 10 프레임마다 루프 클로징을 시도
odometry_loop_weight: 0.1
    - 루프 클로징에 사용할 가중치입니다. 루프 클로징의 영향도를 결정하는 데 사용
odometry_distance_thr: 0.07
    - 정합 과정에서 두 프레임 간 거리 차이의 허용 임계값을 설정
    - 이 값보다 큰 차이가 발생하면 정합을 실패로 간주할 수 있습니다.
    - TODO
icp_method: colored
    - colored는 컬러 정보를 활용한 ICP를 사용
icp_voxelsize: 0.05
    - ICP 정합을 수행할 때 사용할 부피의 크기(복셀 크기)를 설정
    - TODO
icp_distance_thr: 0.07
    - ICP 정합에서 두 점 간의 최대 허용 거리 차이를 설정
    - 이 값보다 차이가 크면 정합에 사용되지 않습니다.
global_registration_method: ransac
    - 전역 정합 방법을 지정합니다. 여기서는 ransac 알고리즘을 사용하여 정합을 수행
registration_loop_weight: 0.1
    - 전역 정합에서 루프 클로징의 가중치를 설정
integrate_color: true
voxel_size: 0.0058
    - 재구성에 사용할 복셀 크기를 설정합니다. 이 값은 부피의 크기를 결정하며, 단위는 미터(m)
    - TODO
trunc_voxel_multiplier: 8.0
    - 거리 정보를 얼마나 잘라낼지(truncation) 결정하는 계수
    예를 들어, trunc_voxel_multiplier=8이고 voxel_size=0.0058이면 
        잘라낼 거리 값은 8 * 0.0058
block_count: 40000
    - 볼륨에 할당할 최대 복셀 블록 수를 설정
    - TODO
est_point_count: 6000000
    - 예상되는 표면 점의 개수를 설정합니다. 이 값은 시스템의 메모리 할당에 사용
surface_weight_thr: 3.0
    - 포인트 클라우드에서 표면을 추출할 때 사용할 가중치 임계값을 설정
    - 이 값보다 작은 가중치는 표면에 포함되지 않습니다.
        """
        volume, poses = slam(depth_file_names, color_file_names, intrinsic,
                             config)
        print('Saving to {}...'.format(config.path_npz))
        volume.save(config.path_npz)
        save_poses('output.log', poses)
        print('Saving finished')
    else:
        volume = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])
