import time
from typing import Annotated
import numpy as np
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class FigurePoseDetect:
    def __init__(self):
        model_path = 'pose_landmarker_full.task'

        # Configure MediaPipe settings
        BaseOptions = python.BaseOptions
        self.PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        self.PoseLandmarkerResult = vision.PoseLandmarkerResult
        self.annotated_image = []
        VisionRunningMode = vision.RunningMode
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, result_callback=self.print_result)

    # function to return annotated image with pose landmarks on the figure given the result
    # pre: result is valid as it is not empty
    def draw_landmarks(self,result:vision.PoseLandmarkerResult, image: mp.Image) -> np.ndarray:
        
        landmark_list = result.pose_landmarks
        annotated_image = np.copy(image.numpy_view())
        
        if len(landmark_list):
            for idx in range(len(landmark_list)):
                # loop through all 33 keypoints and annotate the image
                landmarks = landmark_list[idx]
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                print(type(landmarks[0].x))
                for idy in range(len(landmarks)):
                
                    pose_landmarks_proto.landmark.append(landmark_pb2.NormalizedLandmark(x = landmarks[idy].x, y = landmarks[idy].y, z = landmarks[idy].z))
        
                # draw landmarks on the image copy
                solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())
        
        return annotated_image

    def print_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        landmarks = result.pose_landmarks
        annotated_image = None
        # Ensure landmarks were actually returned or not
        # This ensures list indexing is successful
        if len(landmarks) != 0:
            
            # draw the pose and display it
            self.annotated_image = self.draw_landmarks(result, output_image)
            print(type(annotated_image))
            #cv2.imshow('Pose overlay', annotated_image)
            
            # Print out normalized landmarks for the nose
            #print('The result is {}'.format(landmarks[0][0]))




