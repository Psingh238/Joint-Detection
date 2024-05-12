import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class FigurePoseDetect:
    # list to identify which MediaPipe/Color tracked joint positions align with the reference model
    # reference model joint positions are indicated by pose_remap index values    
    pose_remap = [-1, -2, 0, -3, 12, 14, 16, -4, 11, 13, 15, 24, 26, 28, 23, 25, 27, -5]

    def __init__(self):
        model_path = 'pose_landmarker_heavy.task'

        # Configure MediaPipe settings
        BaseOptions = python.BaseOptions
        self.PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        self.PoseLandmarkerResult = vision.PoseLandmarkerResult
        VisionRunningMode = vision.RunningMode

        # Define variables that need to be passed to the main driver code
        self.annotated_image = []
        self.full_list = []
        self.full_norm_list = []
        
        # Set all the options that need to be used by the MediaPipe model
        # including pose detection confidence and tracking confidence
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, 
            min_pose_detection_confidence=0.70,
            min_tracking_confidence=0.90,
            result_callback=self.print_result)

    # function to return annotated image with pose landmarks on the figure given the result
    def draw_landmarks(self, result:vision.PoseLandmarkerResult, image: mp.Image) -> np.ndarray:
        
        landmark_list = result.pose_landmarks
        annotated_image = np.copy(image.numpy_view())
        
        if len(landmark_list):
            for idx in range(len(landmark_list)):
                # loop through all 33 keypoints and annotate the image
                landmarks = landmark_list[idx]
                # initialize list for displaying MediaPipe landmarks
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                # Add landmarks to the list
                for idy in range(len(landmarks)):
                    pose_landmarks_proto.landmark.append(landmark_pb2.NormalizedLandmark(x = landmarks[idy].x, y = landmarks[idy].y, z = landmarks[idy].z))
                    
                # draw landmarks on the image copy using MediaPipe utility function
                solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image

    # Define function to remap MediaPipe landmarks to specified landmarks in the reference model
    # Also remaps to the proper coordinate system with z up
    def __remap_landmarks(self, landmark_result):
        
        # Define variables to hold the array of joint data
        full_list = []
        full_norm_list = []
        pose_list = None
        pose_norm_list = None
        
        # World landmarks are predicted joint positions recorded in meters with regards to a central origin point
        # These values are under the assumption that the tracked object (in this case the figure), is the size of a human
        # As a result, the values produced are in scale with a human rather than the figure being measured
        mp_landmarks_list = landmark_result.pose_world_landmarks
        mp_norm_landmarks_list = landmark_result.pose_landmarks
        
        # Variable to hold index number for each specific marker
        index = 0
        if len(mp_landmarks_list):
            #as there are multiple predictions of pose stored in mp_landmarks_list, only the first prediction is chosed
            mp_landmarks = mp_landmarks_list[0]
            mp_norm_landmarks = mp_norm_landmarks_list[0]
            for val in FigurePoseDetect.pose_remap:
                
                # If the joint position is tracked with color, initialize the data points to be 0
                # These are placeholders that will be replaced once color joint position data is calculated
                if val < 0:
                    pose_list = [index, 0.0, 0.0, 0.0]
                    pose_norm_list = [index, 0.0, 0.0, 0.0]
                    full_list.append(pose_list)
                    full_norm_list.append(pose_norm_list)
                    
                # If not a color tracked joint position, add joint position data to the appropriate list
                else:
                    # when storing data, remap to align with Boeing coordinate system y = z, z = -y

                    # pose_list stores world landmark data (meters)
                    pose_list = [index, mp_landmarks[val].x, mp_landmarks[val].z, -(mp_landmarks[val].y)]
                    # pose_norm_list stores normalized data
                    pose_norm_list = [index, mp_norm_landmarks[val].x, mp_norm_landmarks[val].z, -(mp_norm_landmarks[val].y)]
                    # append world landmark data to full list
                    full_list.append(pose_list)
                    # append normalized landmark data to full norm list
                    full_norm_list.append(pose_norm_list)
                
                index += 1
        
        return full_list, full_norm_list

    # Callback function for MediaPipe that is continuously called to poll new landmark data
    # Handles drawing of landmarks on image and partially filling up the two arrays with joint data
    def print_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        landmarks = result.pose_landmarks
    
        # Ensure landmarks were actually returned or not
        # This ensures list indexing is successful
        if len(landmarks) != 0:
            #retrieve both world and normalized landmarks and store in object variables
            self.full_list, self.full_norm_list = self.__remap_landmarks(result)
            
            #track left shoulder to indicate that user knows which direction mediapipe assumes the figure is facing
            left_shoulder = landmarks[0][11]
  
            # draw the pose on the given image
            self.annotated_image = self.draw_landmarks(result, output_image)
            # draw reference point (left shoulder) to show that figure is being captured facing the correct way
            self.annotated_image = cv2.circle(self.annotated_image,[int(left_shoulder.x * 640), int(left_shoulder.y * 480)],10,[0, 0, 255],5)




