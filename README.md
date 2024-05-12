# RealSenseAngleDetection

## Project Overview
This project aims to capture pose of a small scale 3D figure utilizing both depth information polled from a depth camera as well as the MediaPipe machine learning model to predict key joint positions to fully capture pose.

## Setup Procedure
Please download the most recent release of this program in the releases tab. This will be in the form of a .zip file that will need to be extracted on the computer. Once extracted, the user will need to click the .exe file present within the release to run the program. The Intel RealSense D405 depth camera will need to be connected to the computer for the program to run. 

## Joint Position data details
The output of this program will produce 18 markers capturing the pose of the figure recorded in meters. These points are in reference to an origin point which is placed in between the hips. Each joint position will be denoted by a marker number when captured.

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