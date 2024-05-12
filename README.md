# RealSenseAngleDetection

## Project Overview
This project aims to capture pose of a small scale 3D figure utilizing both depth information polled from a depth camera as well as the MediaPipe machine learning model to predict key joint positions to fully capture pose.

## Setup Procedure
Please download the most recent release of this program in the releases tab. This will be in the form of a .zip file that will need to be extracted on the computer. Once extracted, the user will need to click the .exe file present within the release to run the program. The Intel RealSense D405 depth camera will need to be connected to the computer for the program to run. 

## Joint Position data details
The output of this program will produce 18 markers capturing the pose of the figure recorded in meters. These points are in reference to an origin point which is placed in between the hips. Each joint position will be denoted by a marker number when captured. These joint positions corroborate with the following reference model.

![image](https://github.com/Psingh238/RealSenseAngleDetection/assets/97202987/8644f00a-050b-4476-a6bc-83ceb10ef916)


## Tracking Joint Positions Through Color
Due to the MediaPipe model not providing the recommended number of joints to fully capture human pose, several joint positions were tracked through the use of color. This was achieved through the use of the OpenCV library in conjunction with the Intel RealSense source development kit to obtain these positions in 3D space. Each color is tracked using masks defined as such: 

```
lower_red = np.array([168, 100, 100])
upper_red = np.array([179, 255,255])
```
where the values are in the order of Hue, Saturation, Value.

To retrieve coordinates using these masks, contours are first created capturing all instances of the specified color, which is defined as follows:
```
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
The program then finds the largest instance of the color and calculates the center of this instance based off of a bounding box super imposed over where the color is. The code to retrieve the center is as follows:
```
x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])

cv2.circle(color_image, (int(x+(w/2)),int(y+(h/2))), 10, color, 2)
color_depth = rs.depth_frame.get_distance(d_frame, int(x+(w/2)), int(y+(h/2)))
center_color = [float(x+(w/2)),float(y+(h/2)), color_depth]
cv2.drawMarker(color_image, (int(center_color[0]), int(center_color[1])), color, cv2.MARKER_CROSS, 20, 3)
```

## Tracking Joint Positions Through MediaPipe
Tracking joint positions through MediaPipe uses a pretrained model developed by google to assume where key joint positions are. The joint positions provided include both normalized coordinates, where the x and y coordinates are pixel values normalized by the image width and length respectively. The other joint position format is world landmarks, which provides the distance from an origin point, the center of the hip, for the x, y, and z coordinates. In this program, both types of data format are utilized, with the normalized data being used for visuals and world landmark data being sent to the data retrieval program.

## Converting Data to the Same Format
Due to the different methods taken to capture the joint positions as shown in the reference model, the color tracked joint positions need to be altered to be in the format of the MediaPipe world landmarks. 
The first step taken is to normalize the x and y coordinates to be in the same format as the MediaPipe normalized landmarks. This is done by dividing the x and y coordinates by the image size as shown below.
```
for color_marker in color_marker_list:
    color_marker[0] = float(color_marker[0] / image_dim[1])
    color_marker[1] = float(color_marker[1] / image_dim[0])
```
In order to align the color tracked joint positions with the MediaPipe world landmarkers, mapping functions and a ratio are used to translate the data.
For the x and y coordinates, mapping functions are used with reference to three world landmarker joint positions: the left shoulder, right shoulder, and the left hip.
Below is how the color data x and y coordinates are translated using this method.
```
left_shoulder_norm = full_pose_norm_dict[8]
left_shoulder_world = full_pose_world_dict[8]

right_shoulder_norm = full_pose_norm_dict[4]
right_shoulder_world = full_pose_world_dict[4]

left_hip_norm = full_pose_norm_dict[14]
left_hip_world = full_pose_world_dict[14]

# remap the x and y values for each color marker
for marker in color_marker_list:
    marker[0] = right_shoulder_world[1] + (left_shoulder_world[1] - right_shoulder_world[1]) * (marker[0] - right_shoulder_norm[1]) / (left_shoulder_norm[1] - right_shoulder_norm[1])
    marker[1] = -left_shoulder_world[3] + ((-left_hip_world[3]) - (-left_shoulder_world[3])) * (marker[1] - (-left_shoulder_norm[3])) / ((-left_hip_norm[3]) - (-left_shoulder_norm[3]))
```
For the z values, a ratio is used to convert the color tracked z value to the MediaPipe world landmarker z value. The ratio is calculated as follows with the left shoulder joint position, provided by MediaPipe, as a reference point.
```
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
           left_shoulder_imageY += random.uniform(-0.05, 0.05)
           left_shoulder_imageX += random.uniform(-0.05, 0.05)
           real_depth = -1.0
           continue
        
    # This ratio is a ratio such that multiplying the RealSense depth value with this ratio would
    # return the depth that MediaPipe has used for all the other landmarks   
    return  left_shoulder_world[2] / real_depth
```
