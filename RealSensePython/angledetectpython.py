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
import FigurePoseDetect
import socket


# Function definitions

# Function to calculate the necessary conversion ratio between real-life meter measurements and MediaPipe meter depth values
# pre: MediaPipe has calculated keypoints and the pose dict is populated with normalized coordinates
#      

def conversion_ratio(full_pose_dict, world_landmarks, image_dim, depth_frame):
    # Get both normalized and world landmarks for the left shoulder
    left_shoulder = full_pose_dict[8]
    left_shoulder_world = world_landmarks[8]
    
    # Prior coordinate translation needs to be reversed
    left_shoulder_imageX = left_shoulder[1]
    left_shoulder_imageY = -left_shoulder[3]

    # The ratio returned is simply the MediaPipe depth at a certain keypoint compared with the actual depth in meters.
    real_depth = 0.0
    
    while real_depth<=0.0:
       try:
           real_depth = rs.depth_frame.get_distance(depth_frame, int(left_shoulder_imageX * image_dim[1]), int(left_shoulder_imageY * image_dim[0]))
       except:
           left_shoulder_imageY += 0.05
        
    # This ratio is a ratio such that multiplying the RealSense depth value with this ratio would
    # return the depth that MediaPipe has used for all the other landmarks   
    return  left_shoulder_world[2] / real_depth


# Function to remap the color markers into the same ranges as the MediaPipe world landmarker
# pre: color_marker_list is a list containing centers of the color markers normalized with respect to the image dimensions

def remap_ranges(color_marker_list, full_pose_norm_dict, full_pose_world_dict):
    # Get coordinate data for both shoulders and one hip point in both dicts
    left_shoulder_norm = full_pose_norm_dict[8]
    left_shoulder_world = full_pose_world_dict[8]
    
    right_shoulder_norm = full_pose_norm_dict[4]
    right_shoulder_world = full_pose_world_dict[4]
    
    left_hip_norm = full_pose_norm_dict[14]
    left_hip_world = full_pose_world_dict[14]
    
    og_y = color_marker_list[0][0]

    # remap the x-values for each color marker
    for marker in color_marker_list:
        marker[0] = right_shoulder_world[1] + (left_shoulder_world[1] + right_shoulder_world[1]) * (marker[0] - right_shoulder_norm[1]) / (left_shoulder_norm[1] - right_shoulder_norm[1])
    
    # remap the y-values for each color marker
    color_marker_list[0][1] = -left_hip_world[3] + ((-left_hip_norm[3]) + (-left_hip_world[3])) * (color_marker_list[0][1] - (-left_shoulder_norm[3])) / ((-left_hip_norm[3]) - (-left_shoulder_norm[3]))

    print(f'Original y is {og_y}, left shoulder ynorm is {-left_shoulder_norm[3]}')
    print(f'New y is {color_marker_list[0][1]}, left shoulder yworld is {-left_shoulder_world[3]}')
    print(f'Left hip yworld is {-left_hip_world[3]}')


# Function to normalize the x and y coordinates of the color markers similar to MediaPipe Model

def normalize_coords(color_marker_list, image_dim):
    for color_marker in color_marker_list:
        color_marker[0] = float(color_marker[0] / image_dim[1])
        color_marker[1] = float(color_marker[1] / image_dim[0])
    
    return color_marker_list

# Function that figures out the contours for a particular color and makes a bounding box on the image for it.
def draw_bound_box(color, color_contour, color_image, d_frame):
    max_area_color = -1
    largest_contour_index_color = -1
    min_area = 50.0
    max_area = 300
    center_color = None
    for i in range (0, len(color_contour)):
        cnt = color_contour[i]
        
        area = cv2.contourArea(cnt)
        if(area > max_area_color and area > min_area and area < max_area):
            max_area_color = area
            largest_contour_index_color = i
    
    if(largest_contour_index_color != -1):
        x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])
        
        cv2.circle(color_image, (int(x+(w/2)),int(y+(h/2))), 10, color, 2)
        color_depth = rs.depth_frame.get_distance(d_frame, int(x+(w/2)), int(y+(h/2)))
        center_color = [float(x+(w/2)),float(y+(h/2)), color_depth]
        cv2.drawMarker(color_image, (int(center_color[0]), int(center_color[1])), color, cv2.MARKER_CROSS, 20, 3)
    
    return center_color

# Function used to calculate co-planar angle given 3 points in 3D space
        
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

# Helper function for calculating co-planar angle          
        
def get_magnitude(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    return math.sqrt(pow(x, 2)+pow(y, 2)+pow(z, 2))

# Function to normalize the lighting of the image

def normalize_color(old_img):
    lab = cv2.cvtColor(old_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

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
profile = pipeline.start(config)

sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
sensor.set_option(rs.option.exposure, 42000)

# define lower and upper bounds for each color
# These are organized as Hue, Saturation, and Value
# Hue goes from 0 deg to 180 deg while Saturation and Value goes from 0 to 255

#lower mid torso [0]
lower_red = np.array([168, 100, 100])
upper_red = np.array([179, 255,255])
#upper mid torso [1]
lower_blue = np.array([101,100,100])
upper_blue = np.array([140, 255, 255])
#right mid shoulder [3]
lower_pink = np.array([140, 100, 100])
upper_pink = np.array([170, 255, 255])
#left mid shoulder [7]
lower_green = np.array([40, 60, 20])
upper_green = np.array([80, 255, 255])
#neck base [17]
lower_teal = np.array([81, 70, 20])
upper_teal = np.array([100, 255, 255])

#ex: http://www.exampledomain.com:8080
api_url = input("Enter API URL for data transmission: ")

# Parse API URL into hostname and port number
parts = api_url.split(':')
hostname = parts[0]
port_num = int(parts[1])

print(hostname, port_num)
'''
# configure client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((hostname, port_num))
'''
try:
    
    with fpd.PoseLandmarker.create_from_options(fpd.options) as landmarker:
        
        # Take time for later comparison
        start_time = time.time()
        write_count = 1
        ratio = -1.0
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
            #color_image = normalize_color(color_image)
            
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            vlim = 30
            v[v<vlim] = 0
            v[v>=vlim] -= 30
            
            hsv_image = cv2.merge((h,s,v))
            color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            #hsv_image = normalize_color(hsv_image)
        
            # These functions create the necessary masks for each color
            mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
            mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
            mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
            mask_teal = cv2.inRange(hsv_image, lower_teal, upper_teal)
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        
            # These functions update the mask and add a blur to them to reduce jittering
            mask_red = cv2.medianBlur(mask_red, 3)
            mask_blue = cv2.medianBlur(mask_blue, 3)
            mask_pink = cv2.medianBlur(mask_pink, 3)
            mask_teal = cv2.medianBlur(mask_teal, 3)
            mask_green = cv2.medianBlur(mask_green, 3)
            
            # Here, all the contours are calculated from the masks
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_teal, _ = cv2.findContours(mask_teal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # The functions draw an appropriately colored rectange and also returns 3D coordinates crossed
            # between the 2D image coordinates the 3D depth data for the center of the rectangles
            center_red = draw_bound_box((0, 0, 255), contours_red, color_image, depth_frame)
            center_blue = draw_bound_box((255, 0, 0), contours_blue, color_image, depth_frame)
            center_pink = draw_bound_box((203, 192, 255), contours_pink, color_image, depth_frame)
            center_teal = draw_bound_box((128, 128, 0), contours_teal, color_image, depth_frame)
            center_green = draw_bound_box((0, 255, 0), contours_green, color_image, depth_frame)
            
            center_list = [center_red, center_blue, center_pink, center_green, center_teal]
            
            
            colors_found = True
            for color in center_list:
                if color == None:
                    colors_found = False
                
            if len(fpd.full_dict) == 18 and colors_found:
                
                # Figure out conversion ratio
                # teal marker
                center_list[4][0] = (fpd.full_dict[8][1]+fpd.full_dict[4][1])/2
                center_list[4][1] = ((-fpd.full_dict[8][3])+(-fpd.full_dict[4][3]))/2
                print(center_list[4][1])
                #pink marker
                center_list[3][0] = (fpd.full_dict[8][1]+center_list[4][0])/2
                center_list[3][1] = ((-fpd.full_dict[8][3])+(center_list[4][1]))/2
                #green marker
                center_list[2][0] = (fpd.full_dict[4][1]+center_list[4][0])/2
                center_list[2][1] = ((-fpd.full_dict[4][3])+(center_list[4][1]))/2
                #red marker
                center_list[0][0] = (center_list[4][0]+((fpd.full_dict[8][1]+fpd.full_dict[4][1])/2))/2
                center_list[0][1] = center_list[4][1]/3
                #blue marker
                center_list[1][0] = (center_list[4][0]+center_list[0][0])/2
                center_list[1][1] = 2*(center_list[4][1]/3)
                
                if(ratio == -1.0):
                    ratio = conversion_ratio(fpd.full_dict, fpd.full_norm_dict, depth_colormap_dim, depth_frame)
                
                '''
                for marker in range(len(fpd.full_dict)):
                    marker_depth = rs.depth_frame.get_distance(depth_frame, int(fpd.full_dict[marker][0]*640), int(-(fpd.full_dict[marker][3])*480))
                    
                    if(fpd.full_dict[marker][2]>10*ratio*marker_depth) and fpd.full_dict[marker][2]>0:
                        fpd.full_dict[marker][2] = ratio*marker_depth
                    elif(fpd.full_dict[marker][2]<10*ratio*marker_depth) and fpd.full_dict[marker][2]<0:
                        fpd.full_dict[marker][2] = ratio*marker_depth
                    else:
                        continue
                '''  
                # normalize color coordinates
                center_list = normalize_coords(center_list, color_colormap_dim)
                
                for marker in range(len(fpd.pose_remap)):
                    if(fpd.pose_remap[marker] < 0):

                        color_index = -(fpd.pose_remap[marker]) - 1
                        '''
                        pose_dict = {
                            'marker': marker,
                            'x': center_list[color_index][0],
                            'y': center_list[color_index][2] * ratio,
                            'z': -(center_list[color_index][1])
                        }
                        '''
                        pose_dict = [marker, center_list[color_index][0], center_list[color_index][2]*ratio, -(center_list[color_index][1])]
                        fpd.full_dict[marker] = pose_dict
            
            #transmits data
            
            if colors_found and (len(fpd.annotated_image) != 0):
                
                remap_ranges(center_list, fpd.full_norm_dict, fpd.full_dict)
                
                #transmit data
                print("transmitting")
                data_info = ['marker', 'x', 'y', 'z']
                csv_data = ''
                for marker in fpd.full_dict:
                    csv_data += ','.join(map(str, marker))
                    csv_data += '\r\n'
                #client_socket.sendall(csv_data.encode())
                #print(csv_data)
                
                if write_count%150==0:
                    write_count = 1
                    with open('joint_data.txt', 'w') as text_file: 
                        print('writing')    
                        text_file.write(csv_data)
            
            #angle = elbow_angle(center_green, center_pink, center_orange)
            
            if len(fpd.annotated_image) != 0:
                #dst = cv2.addWeighted(color_image, 1, fpd.annotated_image, 0.7, 0)
                cv2.imshow('Mediapipe', fpd.annotated_image)
                #cv2.imshow('blended', dst)
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            write_count+=1
            cv2.waitKey(1)

finally:
    # Send stop code to server to signal to it that all requests are handled and close client socket
    #client_socket.sendall(b'\0')
    #client_socket.close()
    # Stop streaming
    pipeline.stop()
