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
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum
import sys
import rclpy
import traceback
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD
from sensor_msgs.msg import CompressedImage, CameraInfo
import re
from cv_bridge import CvBridge

sys.path.append(abspath(__file__))

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


def main():
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
        type = bool,
        default = True,
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
    # ROS2 초기화
    rclpy.init()
    ros2_bridge = None
    try:
        ros2_bridge = RealsenseNode(args)
        rclpy.spin(ros2_bridge)
    except:
        traceback.print_exc()
        if ros2_bridge is not None:
            ros2_bridge.wrap_up()
            ros2_bridge.destroy_node()
            rclpy.shutdown()


    # if sum(o is not False for o in vars(args).values()) != 2:
    #     parser.print_help()
    #     print(f"Please select one of the optional arguments, but not multiple")
    #     exit()
class RealsenseNode(Node):

    def __init__(self, args: argparse.Namespace):
        super().__init__('ros2_bridge')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(RGBD, '/realsense0/rgbd', 10)
        path_output = args.output_folder
        path_depth = join(args.output_folder, "depth")
        path_color = join(args.output_folder, "color")
        if args.record_imgs:
            make_clean_folder(path_output)
            make_clean_folder(path_depth)
            make_clean_folder(path_color)

        # Create a pipeline
        pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        if args.record_imgs or args.record_rosbag:
            # 참고: 640 x 480의 깊이 해상도를 사용하면 부드러운 깊이 경계를 생성할 수 있습니다.
            #       OpenCV 기반의 이미지 시각화를 위해 색상 이미지 형식으로 rs.format.bgr8 을 사용합니다
            config.enable_stream(rs.stream.depth, args.width, args.height,
                                 rs.format.z16, args.fps)

            # 컬러 스트림 설정
            config.enable_stream(rs.stream.color, args.width, args.height,
                                 rs.format.bgr8, args.fps)
        ##############################
        context = rs.context()
        while True:
            connected_devices = [d for d in context.query_devices()]
            print(f"Found {len(connected_devices)} devices")
            usb_port_id = '10-4' #'4-1'
            selected_device = None
            for device in connected_devices:
                physical_port = device.get_info(rs.camera_info.physical_port)
                print("physical_port:", device.get_info(rs.camera_info.physical_port))
                port_id = self.extract_port_id(physical_port)
                print("port_id:", port_id)
                if port_id == usb_port_id:
                    selected_device = device
                    print(
                        f"Device {device.get_info(rs.camera_info.name)} found at {usb_port_id}"
                    )
                    break
            if selected_device is None:
                print("No device found at the specified USB port.")
            else:
                break

        config.enable_device(
            selected_device.get_info(rs.camera_info.serial_number))

        ##############################
        # Start streaming
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        # Using preset HighAccuracy for recording
        if args.record_rosbag or args.record_imgs:
            depth_sensor.set_option(rs.option.visual_preset,
                                    Preset.HighAccuracy)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_scale = depth_sensor.get_depth_scale()
        print("depth_scale: ", depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        try:
            frame_count = 0
            while True:
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                print("frame_count:", frame_count)

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
                    rgbd_msg = RGBD()

                    ## header
                    rgbd_msg.header.stamp = self.get_clock().now().to_msg()
                    rgbd_msg.header.frame_id = 'camera_link'

                    ## rgb camera info
                    rgb_camera_info = CameraInfo()
                    rgb_camera_info.header = rgbd_msg.header
                    rgb_camera_info.height = color_image.shape[0]
                    rgb_camera_info.width = color_image.shape[1]
                    intrinsics = color_frame.profile.as_video_stream_profile(
                    ).intrinsics
                    rgb_camera_info.k = [
                        intrinsics.fx, 0., intrinsics.ppx, 0., intrinsics.fy,
                        intrinsics.ppy, 0., 0., 1.
                    ]
                    rgb_camera_info.height = color_image.shape[0]
                    rgb_camera_info.width = color_image.shape[1]
                    rgbd_msg.rgb_camera_info = rgb_camera_info
                    ## depth camera info
                    depth_camera_info = CameraInfo()
                    depth_camera_info.header = rgbd_msg.header
                    depth_camera_info.height = depth_image.shape[0]
                    depth_camera_info.width = depth_image.shape[1]
                    intrinsics = aligned_depth_frame.profile.as_video_stream_profile(
                    ).intrinsics
                    depth_camera_info.k = [
                        intrinsics.fx, 0., intrinsics.ppx, 0., intrinsics.fy,
                        intrinsics.ppy, 0., 0., 1.
                    ]
                    rgbd_msg.depth_camera_info = depth_camera_info

                    ## depth, color image
                    rgbd_msg.rgb = self.bridge.cv2_to_imgmsg(depth_image,
                                                               encoding='16UC1')
                    rgbd_msg.color = self.bridge.cv2_to_imgmsg(color_image,
                                                               encoding='rgb8')
                    self.publisher.publish(rgbd_msg)
                    # if frame_count == 0:
                    #     save_intrinsic_as_json(
                    #         join(args.output_folder, "camera_intrinsic.json"),
                    #         color_frame)
                    # cv2.imwrite("%s/%06d.png" % \
                    #         (path_depth, frame_count), depth_image)
                    # cv2.imwrite("%s/%06d.jpg" % \
                    #         (path_color, frame_count), color_image)
                    # print("Saved color + depth image %06d" % frame_count)
                    frame_count += 1
        finally:
            pipeline.stop()

    # Function to extract the port ID from the physical port path
    @staticmethod
    def extract_port_id(physical_port):
        match = re.search(r'usb[\d]+/([\d-]+)', physical_port)
        if match:
            return match.group(1)
        return None

if __name__ == "__main__":
    main()
