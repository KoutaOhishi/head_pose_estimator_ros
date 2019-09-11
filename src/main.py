#!/usr/bin/env python
#coding:utf-8

import rospy
import sys
import cv2
import numpy as np
import tf

from multiprocessing import Process, Queue

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion, Vector3
from head_pose_estimator.msg import HeadPose, FaceRect
from cv_bridge import CvBridge, CvBridgeError

class HeadPoseEstimator():
    def __init__(self):
        print "OpenCV version: {}".format(cv2.__version__)
        self.CNN_INPUT_SIZE = 128

        self.mark_detector = MarkDetector()

        height = 480
        width = 640
        self.pose_estimator = PoseEstimator(img_size=(height, width))

        self.pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(8)]

        self.img_queue = Queue()
        self.box_queue = Queue()
        self.box_process = Process(target=self.get_face, args=(self.mark_detector, self.img_queue, self.box_queue,))

        self.process_ctrl_flag = rospy.get_param("process_ctrl", True)
        self.process_ctrl_subscriber = rospy.Subscriber("/head_pose_estimator/process_ctrl", Bool, self.process_ctrl_callback)

        self.sub_img_name = rospy.get_param("sub_img_name", "/usb_cam/image_raw")
        self.img_subscriber = rospy.Subscriber(self.sub_img_name, Image, self.img_callback)

        self.show_result_img_flag = rospy.get_param("show_result_img", True)
        self.show_axis_flag = rospy.get_param("show_axis", True)
        self.show_annotation_box_flag = rospy.get_param("show_annotation_box", True)
        self.pub_result_img_flag = rospy.get_param("pub_result_img", True)

        self.img_publisher = rospy.Publisher("/head_pose_estimator/result_image", Image, queue_size=10)

        self.result_publisher = rospy.Publisher("/head_pose_estimator/head_pose", HeadPose, queue_size=10)

        self.box_process.start()

    def get_face(self, detector, img_queue, box_queue):
        """
        画像キューから顔を取り出します。
        処理はマルチプロセッシングで行います。
        """
        while not rospy.is_shutdown():
            image = img_queue.get()
            box = detector.extract_cnn_facebox(image)
            box_queue.put(box)

    def process_ctrl_callback(self, msg):
        self.process_ctrl_flag = msg.data

    def euler_to_quaternion(self, euler):
        # オイラー角からクォータニオンに変換
        # 鏡写しになるように値を調整
        q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
        return Quaternion(x=q[1], y=-q[2], z=-q[3], w=q[0])
        #return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def img_callback(self, img_msg):
        if self.process_ctrl_flag == True:
            try:
                cv_img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
                flip_frame = cv2.flip(cv_img, 2)  # 画像を鏡写しにする
                frame = cv_img

                img_height, img_width, _ = frame.shape[:3]

                self.img_queue.put(frame)

                facebox = self.box_queue.get()

                # 顔の検出に成功
                if facebox is not None:
                    # 画像から顔をトリミングしてCNNでlandmarksを見つける
                    # img[top : bottom, left : right]
                    face_img = frame[facebox[1]:facebox[3], facebox[0]:facebox[2]]
                    face_img = cv2.resize(face_img, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                    face_rect = FaceRect()
                    face_rect.x = facebox[0]
                    face_rect.y = facebox[1]
                    face_rect.width = facebox[2] - facebox[0]
                    face_rect.height = facebox[3] - facebox[1]

                    #face_rect.x = img_width - facebox[0] - (facebox[2] - facebox[0])
                    #face_rect.y = facebox[1]
                    #face_rect.width = facebox[2] - facebox[0]
                    #face_rect.height = facebox[3] - facebox[1]

                    marks = self.mark_detector.detect_marks([face_img])

                    marks *= (facebox[2] - facebox[0])
                    marks[:, 0] += facebox[0]
                    marks[:, 1] += facebox[1]

                    # Uncomment following line to show raw marks.
                    #self.mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

                    # Uncomment following line to show facebox.
                    #self.mark_detector.draw_box(frame, [facebox])

                    pose = self.pose_estimator.solve_pose_by_68_points(marks)
                    #rotation = pose[0]
                    #euler = Vector3(x=rotation[0], y=rotation[1], z=rotation[2])
                    #quaternion = self.euler_to_quaternion(euler)

                    # stabilize the pose
                    steady_pose = []
                    pose_np = np.array(pose).flatten()
                    for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                        ps_stb.update([value])
                        steady_pose.append(ps_stb.state[0])
                    steady_pose = np.reshape(steady_pose, (-1, 3))

                    rotation = steady_pose[0]
                    euler = Vector3(x=rotation[0], y=rotation[1], z=rotation[2])
                    head_rotation = self.euler_to_quaternion(euler)
                    #print head_rotation
                    #print "---"

                    head_pose = HeadPose()
                    head_pose.face_rect = face_rect
                    head_pose.head_rotation = head_rotation
                    self.result_publisher.publish(head_pose)

                    # Uncomment following line to draw pose annotation on frame.
                    #self.pose_estimator.draw_annotation_box(frame, pose[0], pose[1], color=(255, 128, 128))

                    # Uncomment following line to draw stabile pose annotation on frame.
                    if self.show_annotation_box_flag == True:
                        self.pose_estimator.draw_annotation_box(frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

                    # Uncomment following line to draw head axes on frame.
                    # openCV3.4.0では動かない
                    #self.pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

                    if self.show_axis_flag == True:
                        self.pose_estimator.draw_axis(frame, steady_pose[0], steady_pose[1])


                if self.show_result_img_flag == True:
                    cv2.imshow("result", frame)
                    cv2.waitKey(1)

                if self.pub_result_img_flag == True:
                    try:
                        pub_img = CvBridge().cv2_to_imgmsg(frame, "bgr8")
                        self.img_publisher.publish(pub_img)

                    except CvBridgeError, e:
                        rospy.logerr(str(e))

            except CvBridgeError, e:
                rospy.logerr(str(e))


if __name__ == "__main__":
    rospy.init_node("head_pose_estimator", anonymous=False)
    hpe = HeadPoseEstimator()
    rospy.spin()
