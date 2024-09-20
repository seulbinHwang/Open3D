# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum

import sys

sys.path.append(abspath(__file__))
from realsense_helper import get_profiles

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()


def save_intrinsic_as_json(filename, frame):
    """카메라의 내부 보정 정보를 JSON 파일로 저장합니다. """
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        # TODO: depth_scale 도 저장해야 하는지?
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")

    # 기존 인자 파서 부분에 추가
    parser.add_argument("--output_folder",
                        default='../dataset/realsense/',
                        help="set output folder")
    parser.add_argument("--record_rosbag",
                        action='store_true',
                        help="Recording rgbd stream into realsense.bag")
    parser.add_argument(
        "--record_imgs",
        action='store_true',
        help="Recording save color and depth images into realsense folder")
    parser.add_argument("--playback_rosbag",
                        action='store_true',
                        help="Play recorded realsense.bag file")

    # 새로운 인자 추가: 해상도 및 프레임 속도 설정
    parser.add_argument("--width",
                        type=int,
                        default=640,
                        help="Width of the frames")
    parser.add_argument("--height",
                        type=int,
                        default=480,
                        help="Height of the frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--max_depth",
                        type=float,
                        default=3.0,
                        help="Maximum depth value in meters to trust")
    args = parser.parse_args()
    # if sum(o is not False for o in vars(args).values()) != 2:
    #     parser.print_help()
    #     print(f"Please select one of the optional arguments, but not multiple")
    #     exit()

    path_output = args.output_folder
    path_depth = join(args.output_folder, "depth")
    path_color = join(args.output_folder, "color")
    if args.record_imgs:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    path_bag = join(args.output_folder, "realsense.bag")
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # color_profiles, depth_profiles = get_profiles()

    if args.record_imgs or args.record_rosbag:
        # 참고: 640 x 480의 깊이 해상도를 사용하면 부드러운 깊이 경계를 생성할 수 있습니다.
        #       OpenCV 기반의 이미지 시각화를 위해 색상 이미지 형식으로 rs.format.bgr8 을 사용합니다
        # for a_color_profile, a_depth_profile in zip(color_profiles, depth_profiles):
        #     print('Using the profiles: \n  color:{}, depth:{}'.format(
        #         a_color_profile, a_depth_profile))
        # print('Using the default profiles: \n  color:{}, depth:{}'.format(
        #     color_profiles[0], depth_profiles[0]))
        # w, h, fps, fmt = depth_profiles[0]
        # config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        # w, h, fps, fmt = color_profiles[0]
        # config.enable_stream(rs.stream.color, w, h, fmt, fps)
        # 깊이 스트림 설정
        config.enable_stream(rs.stream.depth, args.width, args.height,
                             rs.format.z16, args.fps)

        # 컬러 스트림 설정
        config.enable_stream(rs.stream.color, args.width, args.height,
                             rs.format.bgr8, args.fps)

        if args.record_rosbag:
            config.enable_record_to_file(path_bag)
    if args.playback_rosbag:
        config.enable_device_from_file(path_bag, repeat_playback=True)
    ##############################
    context = rs.context()
    while True:
        connected_devices = [d for d in context.query_devices()]
        print(f"Found {len(connected_devices)} devices")
        usb_port_id = '4-1'
        selected_device = None
        for device in connected_devices:
            if device.get_info(rs.camera_info.physical_port) == usb_port_id:
                selected_device = device
                print(
                    f"Device {device.get_info(rs.camera_info.name)} found at {usb_port_id}"
                )

        if selected_device is None:
            print("No device found at the specified USB port.")
        else:
            print("Device found")
            break

    config.enable_device(selected_device.get_info(rs.camera_info.serial_number))

    ##############################
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()
    print("depth_scale: ", depth_scale)

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    # clipping_distance_in_meters = args.max_depth  # 3 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    frame_count = 0
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if args.record_imgs:
                if frame_count == 0:
                    save_intrinsic_as_json(
                        join(args.output_folder, "camera_intrinsic.json"),
                        color_frame)
                cv2.imwrite("%s/%06d.png" % \
                        (path_depth, frame_count), depth_image)
                cv2.imwrite("%s/%06d.jpg" % \
                        (path_color, frame_count), color_image)
                print("Saved color + depth image %06d" % frame_count)
                frame_count += 1

            # # Remove background - Set pixels further than clipping_distance to grey
            # grey_color = 153
            # #depth image is 1 channel, color is 3 channels
            # depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            # bg_removed = np.where((depth_image_3d > clipping_distance) | \
            #         (depth_image_3d <= 0), grey_color, color_image)
            #
            # # Render images
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            # images = np.hstack((bg_removed, depth_colormap))
            # cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Recorder Realsense', images)
            # key = cv2.waitKey(1)
            #
            # # if 'esc' button pressed, escape loop and exit program
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     break
    finally:
        pipeline.stop()
