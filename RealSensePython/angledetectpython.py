#    Copyright 2023-2024 Seattle University Team ECE 24.2
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import time
import random
import pyrealsense2 as rs
import numpy as np
import cv2
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
    
    while real_depth <= 0.0:
       try:
           real_depth = rs.depth_frame.get_distance(depth_frame, int(left_shoulder_imageX * image_dim[1]), int(left_shoulder_imageY * image_dim[0]))
       except Exception as e:
           # Shift the coordinate around by a random amount until the get_distancee functions works
           left_shoulder_imageY += random.uniform(-0.05, 0.05)
           left_shoulder_imageX += random.uniform(-0.05, 0.05)
           real_depth = -1.0
           continue
        
    # This ratio is a ratio such that multiplying the RealSense depth value with this ratio would
    # return the depth that MediaPipe has used for all the other landmarks   
    return  left_shoulder_world[2] / real_depth


# Function to remap the color markers into the same ranges as the MediaPipe world landmarker
# pre: color_marker_list is a list containing centers of the color markers normalized with respect to the image dimensions

def remap_ranges(color_marker_list, full_pose_norm_dict, full_pose_world_dict):
    
    # Get coordinate data for both shoulders and one hip point in both dicts
    # Note: normalized landmarks are smaller when considering the right of the 3D printed figure and the top of the figure
    # For the world landmarks, it is the same case; however, the y-origin is at the center of the hip instead.
    left_shoulder_norm = full_pose_norm_dict[8]
    left_shoulder_world = full_pose_world_dict[8]
    
    right_shoulder_norm = full_pose_norm_dict[4]
    right_shoulder_world = full_pose_world_dict[4]
    
    left_hip_norm = full_pose_norm_dict[14]
    left_hip_world = full_pose_world_dict[14]
    
    # remap the x and y values for each color marker using mapping functions
    for marker in color_marker_list:
        marker[0] = right_shoulder_world[1] + (left_shoulder_world[1] - right_shoulder_world[1]) * (marker[0] - right_shoulder_norm[1]) / (left_shoulder_norm[1] - right_shoulder_norm[1])
        marker[1] = -left_shoulder_world[3] + ((-left_hip_world[3]) - (-left_shoulder_world[3])) * (marker[1] - (-left_shoulder_norm[3])) / ((-left_hip_norm[3]) - (-left_shoulder_norm[3]))

# Function to normalize the x and y coordinates of the color markers similar to MediaPipe normalized landmarks

def normalize_coords(color_marker_list, image_dim):
    #normalization factor used for mediapipe normalized landmarkers are the image dimensions
    for color_marker in color_marker_list:
        color_marker[0] = float(color_marker[0] / image_dim[1])
        color_marker[1] = float(color_marker[1] / image_dim[0])
    
    return color_marker_list

# Function that figures out the contours for a particular color and makes a bounding box on the image for it.
def draw_bound_box(color, color_contour, color_image, d_frame):
    max_area_color = -1
    largest_contour_index_color = -1
    # size boundaries to not track objects too small or too big
    min_area = 50.0
    max_area = 300
    #variable to store center point of color marker
    center_color = None
    # go through all contours of the specific color
    for i in range (0, len(color_contour)):
        cnt = color_contour[i]
        
        area = cv2.contourArea(cnt)
        # check to track largest presence of color in image capture that fits within the specified size bounds
        if(area > max_area_color and area > min_area and area < max_area):
            max_area_color = area
            #track which contour has the largest area within the bounds stated before

            largest_contour_index_color = i
    # if the color was found, modify the image stream to draw the marker of the respective color and identify the center of the marker
    if(largest_contour_index_color != -1):
        x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])
        # draw circle around the center in the image using the color being tracked
        cv2.circle(color_image, (int(x + (w / 2)), int(y + (h / 2))), 10, color, 2)
        
        # poll depth using depth frame and xy coordinates of the circle
        color_depth = rs.depth_frame.get_distance(d_frame, int(x + (w / 2)), int(y + (h / 2)))

        # store all coordinate values in the list
        center_color = [float(x + (w / 2)), float(y + (h / 2)), color_depth]
        
        #highlight the exact position of the color tracked joint position by drawing a cross of the same color
        cv2.drawMarker(color_image, (int(center_color[0]), int(center_color[1])), color, cv2.MARKER_CROSS, 20, 3)
    
    return center_color

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

# Function inputs are: stream_type, width, height, data_format, frame_rate
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

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

#ex: www.exampledomain.com:8080
api_url = input("Enter API URL for data transmission (example: <hostname>:8080): ")

# Parse API URL into hostname and port number
parts = api_url.split(':')
hostname = parts[0]
port_num = int(parts[1])

# configuring socket connection to server endpoint

# configure client socket and connect to server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((hostname, port_num))

# Print message to let user know how to exit the program
print('Hit Ctrl + C to quit program')

try:
    
    with fpd.PoseLandmarker.create_from_options(fpd.options) as landmarker:
        
        # Take time for later comparison
        start_time = time.time()
        ratio = -1
        while True:        
                            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Loop around if both frames are not available
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

            # Get dimensions of depth and color frame in format (height, width)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            
            # correct image stream to reduce brightness
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            vlim = 30
            v[v<vlim] = 0
            v[v>=vlim] -= 30
            
            hsv_image = cv2.merge((h,s,v))
            color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            
        
            # These functions create the necessary masks for each color
            mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
            mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
            mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
            mask_teal = cv2.inRange(hsv_image, lower_teal, upper_teal)
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            
            mask_list = [mask_red, mask_blue, mask_pink, mask_teal, mask_green]
            
            # These functions update the mask and add a blur to them to reduce jittering
            for mask in mask_list:
                mask = cv2.medianBlur(mask, 3)
                
            # Here, all the contours are calculated from the masks
            contours_red, _ = cv2.findContours(mask_list[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_blue, _ = cv2.findContours(mask_list[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pink, _ = cv2.findContours(mask_list[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_teal, _ = cv2.findContours(mask_list[3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_list[4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # The functions draw an appropriately colored rectange and also returns 3D coordinates crossed
            # between the 2D image coordinates the 3D depth data for the center of the rectangles
            center_red = draw_bound_box((0, 0, 255), contours_red, color_image, depth_frame)
            center_blue = draw_bound_box((255, 0, 0), contours_blue, color_image, depth_frame)
            center_pink = draw_bound_box((203, 192, 255), contours_pink, color_image, depth_frame)
            center_teal = draw_bound_box((128, 128, 0), contours_teal, color_image, depth_frame)
            center_green = draw_bound_box((0, 255, 0), contours_green, color_image, depth_frame)
            
            center_list = [center_red, center_blue, center_pink, center_green, center_teal]
            
            # in order to make color data have the same units as mediapipe, all color tracked joint positions must be found
            colors_found = True
            for color in center_list:
                if color == None:
                    colors_found = False

            # checks if all joint positions were found (color and MediaPipe)    
            if (len(fpd.full_list) == 18 and 
                len(fpd.full_norm_list) == 18 and 
                colors_found):

                if(ratio == -1):
                    ratio = conversion_ratio(fpd.full_norm_list, fpd.full_list, depth_colormap_dim, depth_frame)
                
                # normalize color coordinates
                center_list = normalize_coords(center_list, color_colormap_dim)
                
                # remap to world landmark coordinate space
                remap_ranges(center_list, fpd.full_norm_list, fpd.full_list)
                # go through list storing all joint positions
                for marker in range(len(fpd.pose_remap)):
                    # check if the joint position is a color tracked joint (marker is negative)
                    if(fpd.pose_remap[marker] < 0):
                        # translate marker to where the color tracked joint data is located in center_list
                        color_index = -(fpd.pose_remap[marker]) - 1
                        # Assign the appropriate joint position with the data stored in center list
                        # when assigning, modify the depth value with the calculated ratio between the color and MediaPipe depth values
                        # when assigning, also remap the coordinates so that y = z and z = -y to align with Boeing's coordinate system
                        pose_dict = [marker, center_list[color_index][0], center_list[color_index][2]*ratio - 0.025, -(center_list[color_index][1])]
                        fpd.full_list[marker] = pose_dict
            
            # Check whether all joint positions are found. If so, start data transmission
            if colors_found and (len(fpd.annotated_image) != 0):
                
                # notify user that data will be transmitted
                print("Transmitting data")
                csv_data = ''
                
                # convert the numerical data into strings and join them together using commas
                # to create comma-separated values
                for marker in fpd.full_list:
                    csv_data += ','.join(map(str, marker))
                    csv_data += '\r\n'
                
                # data is sent after encoding to the server. Uses default UTF-8 encoding
                client_socket.sendall(csv_data.encode())

            # Show images            
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            # If MediaPipe has tracked joints positions, display another image with overlaid MediaPipe joint positions
            if len(fpd.annotated_image):
                cv2.imshow('Mediapipe', fpd.annotated_image)
            cv2.waitKey(1)

finally:
    # Send stop code to server to signal to it that all requests are handled and close client socket
    client_socket.sendall(b'\0')
    client_socket.close()
    # Stop streaming
    pipeline.stop()
