import numpy as np
import mediapipe as mp
import cv2
import json
import msvcrt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import requests

class FigurePoseDetect:
        
    pose_remap = [-1, -2, 0, -3, 12, 14, 16, -4, 11, 13, 15, 24, 26, 28, 23, 25, 27, -5]

    def __init__(self):
        model_path = 'pose_landmarker_heavy.task'

        # Configure MediaPipe settings
        BaseOptions = python.BaseOptions
        self.PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        self.PoseLandmarkerResult = vision.PoseLandmarkerResult
        self.annotated_image = []
        self.full_list = []
        self.full_norm_list = []
        VisionRunningMode = vision.RunningMode
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, 
            min_pose_detection_confidence=0.70,
            min_tracking_confidence=0.90,
            result_callback=self.print_result)

    # function to return annotated image with pose landmarks on the figure given the result
    # pre: result is valid as it is not empty
    def draw_landmarks(self, result:vision.PoseLandmarkerResult, image: mp.Image) -> np.ndarray:
        
        landmark_list = result.pose_landmarks
        annotated_image = np.copy(image.numpy_view())
        
        if len(landmark_list):
            for idx in range(len(landmark_list)):
                # loop through all 33 keypoints and annotate the image
                landmarks = landmark_list[idx]
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                
                for idy in range(len(landmarks)):
                    pose_landmarks_proto.landmark.append(landmark_pb2.NormalizedLandmark(x = landmarks[idy].x, y = landmarks[idy].y, z = landmarks[idy].z))
                    
                # draw landmarks on the image copy
                solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image

    # Define function to remap MediaPipe landmarks to specified landmarks
    # Also remaps to the proper coordinate system with z up
    # Pre: pose_landmarks would be valid due to location of function call
    def __remap_landmarks(self, landmark_result):
        full_list = []
        full_norm_list = []
        pose_list = None
        pose_norm_list = None
        mp_landmarks_list = landmark_result.pose_world_landmarks
        mp_norm_landmarks_list = landmark_result.pose_landmarks
        index = 0
        if len(mp_landmarks_list):
            mp_landmarks = mp_landmarks_list[0]
            mp_norm_landmarks = mp_norm_landmarks_list[0]
            for val in FigurePoseDetect.pose_remap:
                if val < 0:
                    '''
                    pose_dict = {
                        'marker': index,
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0
                        }
                    '''
                    pose_list = [index, 0.0, 0.0, 0.0]
                    pose_norm_list = [index, 0.0, 0.0, 0.0]
                    full_list.append(pose_list)
                    full_norm_list.append(pose_norm_list)
                else:
                    '''    
                    pose_dict = {
                        'marker': index,
                        'x': mp_landmarks[val].z,
                        'y': mp_landmarks[val].x,
                        'z': -(mp_landmarks[val].y)
                        }
                    '''
                    pose_list = [index, mp_landmarks[val].x, mp_landmarks[val].z, -(mp_landmarks[val].y)]
                    pose_norm_list = [index, mp_norm_landmarks[val].x, mp_norm_landmarks[val].z, -(mp_norm_landmarks[val].y)]
                    full_list.append(pose_list)
                    full_norm_list.append(pose_norm_list)
                
                index += 1
        
        return full_list, full_norm_list


    def print_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        landmarks = result.pose_landmarks
    
        # Ensure landmarks were actually returned or not
        # This ensures list indexing is successful
        if len(landmarks) != 0:
            
            self.full_list, self.full_norm_list = self.__remap_landmarks(result)
            left_shoulder = landmarks[0][11]
  
            # draw the pose on given image and return for access outside class
            self.annotated_image = self.draw_landmarks(result, output_image)
            self.annotated_image = cv2.circle(self.annotated_image,[int(left_shoulder.x * 640), int(left_shoulder.y * 480)],10,[0, 0, 255],5)




