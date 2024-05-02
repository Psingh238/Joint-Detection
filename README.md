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