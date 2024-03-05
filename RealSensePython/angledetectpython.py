## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##       OpenCV and Numpy integration        ##
###############################################

import time
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import FigurePoseDetect

# Function definitions

def draw_bound_box(color, color_contour, color_image, d_frame):
    max_area_color = -1
    largest_contour_index_color = -1
    min_area = 100.0
    center_color = None
    for i in range (0, len(color_contour)):
        cnt = color_contour[i]
        
        area = cv2.contourArea(cnt)
        if(area > max_area_color and area > min_area):
            max_area_color = area
            largest_contour_index_color = i
    
    if(largest_contour_index_color != -1):
        x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])
        cv2.rectangle(color_image, (x, y),(x+w, y+h),color,2)
        
        color_depth = rs.depth_frame.get_distance(d_frame, int(x+(w/2)), int(y+(h/2)))
        center_color = (float(x+(w/2)),float(y+(h/2)), color_depth)
        cv2.drawMarker(color_image, (int(center_color[0]), int(center_color[1])), color, cv2.MARKER_CROSS, 20, 3)
    
    return center_color
        
def elbow_angle(forearm, joint, backarm):
    
    if(forearm != None and joint!=None and backarm!=None):
        
        forearm_joint = np.array([forearm[0]-joint[0], forearm[1]-joint[1], forearm[2]-joint[2]])
        backarm_joint = np.array([backarm[0]-joint[0], backarm[1]-joint[1], backarm[2]-joint[2]])
        
        mag_forearm = get_magnitude(forearm_joint)
        mag_backarm = get_magnitude(backarm_joint)
        
        
        forearm_joint/=mag_forearm
        backarm_joint/=mag_backarm
        theta = math.acos(np.dot(forearm_joint, backarm_joint))
        
        return (theta*180)/math.pi
    return -1
          
        
def get_magnitude(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    return math.sqrt(pow(x, 2)+pow(y, 2)+pow(z, 2))

def normalize_color(oldHSV):
    h, s, v = cv2.split(oldHSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    
    oldHSV = cv2.merge([h, s, v])
    return oldHSV

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Create figure pose detection object to easily access common functions
fpd = FigurePoseDetect.FigurePoseDetect()        

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# define lower and upper bounds for each color
# These are organized as Hue, Saturation, and Value
# Hue goes from 0 deg to 180 deg while Saturation and Value goes from 0 to 255
lower_red = np.array([160, 20, 20])
upper_red = np.array([179, 255,255])

lower_green = np.array([40,50,40])
upper_green = np.array([80, 255, 255])
        
lower_pink = np.array([135, 50, 50])
upper_pink = np.array([155, 255, 255])

lower_yellow = np.array([25, 50, 50])
upper_yellow = np.array([35, 255, 255])

lower_green = np.array([36, 50, 70])
upper_green = np.array([89, 255, 255])

lower_orange = np.array([10, 50, 70])
upper_orange = np.array([25, 255, 255])

try:
    
    with fpd.PoseLandmarker.create_from_options(fpd.options) as landmarker:
        
        # Take time for later comparison
        start_time = time.time()
        
        while True:        
                            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert image to MediaPipe image for use with pose landmarker
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
            
            # Generate timestamp in milliseconds for the callback function
            timestamp = int((time.time() - start_time) * 1000)
            
            # Perform landmarking
            landmarker.detect_async(mp_image, timestamp)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
        
            HSVImage = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            HSVImage = normalize_color(HSVImage)
        
            mask_red = cv2.inRange(HSVImage, lower_red, upper_red)
            mask_green = cv2.inRange(HSVImage, lower_green, upper_green)
            mask_pink = cv2.inRange(HSVImage, lower_pink, upper_pink)
            mask_orange = cv2.inRange(HSVImage, lower_orange, upper_orange)
        
            mask_red = cv2.medianBlur(mask_red, 3)
            mask_green = cv2.medianBlur(mask_green, 3)
            mask_pink = cv2.medianBlur(mask_pink, 3)
            mask_orange = cv2.medianBlur(mask_orange, 3)
        
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            center_red = draw_bound_box((0, 0, 255), contours_red, color_image, depth_frame)
            center_green = draw_bound_box((0, 255, 0), contours_green, color_image, depth_frame)
            center_pink = draw_bound_box((255, 0, 255), contours_pink, color_image, depth_frame)
            center_orange = draw_bound_box((255, 255, 0), contours_pink, color_image, depth_frame)
        
            angle = elbow_angle(center_green, center_pink, center_orange)
        
            if len(fpd.annotated_image) != 0:
                #dst = cv2.addWeighted(color_image, 1, fpd.annotated_image, 0.7, 0)
                cv2.imshow('Mediapipe', fpd.annotated_image)
                #cv2.imshow('blended', dst)
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
        
            cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
