# Joint-Detection

## Project Overview
This project aims to capture pose of a small scale 3D figure utilizing both depth information polled from a depth camera as well as the MediaPipe machine learning model to predict key joint positions to fully capture pose.

## Building From Source

Please follow the following steps to build the project from the source code. Please also note the prerequisites.

### Prerequisites

- Python version 3.10.11 (it is highly probable that version 3.10.X in general would work)
- Microsoft Visual Studio 2022 with Python development feature installed

### Setting up the project in Visual Studio

1. Use GitHub's "Open with Visual Studio" feature to open the project in Visual Studio.
2. Create the Python virtual environment in the main folder using the following command in Windows Command Prompt: ``python -m venv venv``.
3. Add the newly created virtual environment to the Visual Studio project by using Visual Studio's Python virtual environment window.
4. Right-click on the virtual environment in the Visual Studio Solution Explorer and click on "install from requirements.txt."
5. All required Python modules are now installed and changes can be made to the code as deemed necessary.

### Compiling to EXE file

1. Start the Python virtual environment in the Command Prompt by using the following command: ``venv\Scripts\activate``. This Command Prompt can be accessed using the integrated Command Prompt in Visual Studio.
2. Install the Python module ``pyinstaller`` using the following command: ``pip install pyinstaller``.
3. There will be two EXE files: the server and the joint tracking software.
    - To compile the server.py into an EXE, run the following command: ``pyinstaller server.py``
    - To compile the joint tracking software into an EXE, run the following command (note that the order of files matters): ``pyinstaller FigurePoseDetect.py driver.py``

    The new EXE files will be available under the dist folder. For the joint tracking EXE to work correctly, the pose_landmarker_heavy.task file needs to be copied into the folder for the joint tracking software next to the associated EXE file.

4. The new EXE files are now ready to be run.

## Setup Procedure For Directly Using EXE Files
Please download the most recent release of this program in the releases tab. This will be in the form of a .zip file that will need to be extracted on the computer. Once extracted, the user will need to click the .exe file for starting the data retrieval program present within the release folder. The user should then click the joint tracking program .exe file. The Intel RealSense D405 depth camera will need to be connected to the computer for the program to run. 

## Joint Position Data Details
The output of this program will produce 22 markers capturing the pose of the figure recorded in meters. These points are in reference to an origin point which is placed in between the hips. Each joint position will be denoted by a marker number when captured. These joint positions corroborate with the following reference model.

![image](https://github.com/Psingh238/RealSenseAngleDetection/assets/97202987/8644f00a-050b-4476-a6bc-83ceb10ef916)


## Tracking Joint Positions Through Color
Due to the MediaPipe model not providing the recommended number of joints to fully capture human pose, several joint positions were tracked through the use of color. This was achieved through the use of the OpenCV library in conjunction with the Intel RealSense source development kit to obtain these positions in 3D space. Each color is tracked using masks defined as such: 

``` python
lower_red = np.array([168, 100, 100])
upper_red = np.array([179, 255,255])
```
where the values are in the order of Hue, Saturation, Value.

To retrieve coordinates using these masks, contours are first created capturing all instances of the specified color, which is defined as follows:
``` python
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
The program then finds the largest instance of the color and calculates the center of this instance based off of a bounding box super imposed over where the color is. The code to retrieve the center is as follows:
``` python
x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])
# draw circle around the center in the image using the color being tracked
cv2.circle(color_image, (int(x + (w / 2)), int(y + (h / 2))), 10, color, 2)

# poll depth using depth frame and xy coordinates of the circle
color_depth = rs.depth_frame.get_distance(d_frame, int(x + (w / 2)), int(y + (h / 2)))

# store all coordinate values in the list
center_color = [float(x + (w / 2)), float(y + (h / 2)), color_depth]

```

## Tracking Joint Positions Through MediaPipe
Tracking joint positions through MediaPipe uses a pretrained model developed by google to assume where key joint positions are. The joint positions provided include both normalized coordinates, where the x and y coordinates are pixel values normalized by the image width and length respectively. The other joint position format is world landmarks, which provides the distance from an origin point, the center of the hip, for the x, y, and z coordinates. In this program, both types of data format are utilized, with the normalized data being used for visuals and world landmark data being sent to the data retrieval program.

## Converting Data to the Same Format
Due to the different methods taken to capture the joint positions as shown in the reference model, the color tracked joint positions need to be altered to be in the format of the MediaPipe world landmarks. 
The first step taken is to normalize the x and y coordinates to be in the same format as the MediaPipe normalized landmarks. This is done by dividing the x and y coordinates by the image size as shown below.
``` python
for color_marker in color_marker_list:
    color_marker[0] = float(color_marker[0] / image_dim[1])
    color_marker[1] = float(color_marker[1] / image_dim[0])
```
In order to align the color tracked joint positions with the MediaPipe world landmarkers, mapping functions and a ratio are used to translate the data.
For the x and y coordinates, mapping functions are used with reference to three world landmarker joint positions: the left shoulder, right shoulder, and the left hip.
Below is how the color data x and y coordinates are translated using this method.
``` python
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
``` python
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
```
Finally, the data will be remapped to match Boeing's coordinate system, where y will be assigned to z and z is assigned to -y.

## Data Visualization

When capturing joint position data, this data will also be displayed to the user to show how the program is tracking the joint positions. This is mainly handled by OpenCV, where it will display two windows to the user, showing the color tracked and the MediaPipe tracked joint positions.

![image](https://github.com/Psingh238/RealSenseAngleDetection/assets/97202987/9994dcb4-f37c-4a3f-a8b8-cbbe6ffe0c07)

### Color Data display
When displaying color tracked data, as highlighted before, several bounding boxes are created for each color tracked joint. These are overlaid on the image using the following code:
``` python
x, y, w, h = cv2.boundingRect(color_contour[largest_contour_index_color])
cv2.circle(color_image, (int(x + (w / 2)),int(y + (h / 2))), 10, color, 2)

```
As shown in the image above, each color joint position has its own color attributed to it, allowing each point to be easily identifiable
### MediaPipe data display
When displaying MediaPipe joint data, the data must first be assigned to a ``pose_landmarks_proto`` list, which is specifically used to draw MediaPipe landmarks, where it takes in normalized MediaPipe landmark data. The definition of this list is defined as such:
``` python
# initialize list for displaying MediaPipe landmarks
pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
# Add landmarks to the list
for idy in range(len(landmarks)):
    pose_landmarks_proto.landmark.append(landmark_pb2.NormalizedLandmark(x = landmarks[idy].x, y = landmarks[idy].y, z = landmarks[idy].z))
```
Once assigned, the data will then be displayed using this function:
``` python
solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())
```
Another piece of information that is added on top of the landmarker data is a highlighted joint to indicate which way MediaPipe thinks the figure is facing. This is added using the following code:
``` python
self.annotated_image = cv2.circle(self.annotated_image,[int(left_shoulder.x * 640), int(left_shoulder.y * 480)],10,[0, 0, 255],5)
```
The highlighted joint in this case is the left shoulder of the figure. By using this indicator, the user can tell whether MediaPipe's prediction of where the figure is facing is accurate or inaccurate immediately.


## Data Transmission

The joint position data is formatted as comma-separated value (CSV) strings where the data format is ``maker_num, x, y, z``. There are 22 rows of data, corresponding to the 22 markers on the reference model. These rows are separated using the ``\r\n`` escape characteers which are also used by the server code to reconstruct the data. A socket connection using TCP/IPv4 is used to send data from the joint tracking program to the server program on port 5000. Note that this port can be changed by changing the source code for the server. This is a real-time connection as the server can continuously parse the incoming data and display it on the console.

### Client-side code

Below is the relevant part of the joint tracking program that is responsible for converting the numerical data into the CSV string to send to the server.

``` python
csv_data = ''
                
# convert the numerical data into strings and join them together using commas
# to create comma-separated values
for marker in fpd.full_list:
    csv_data += ','.join(map(str, marker))
    csv_data += '\r\n'
                
# data is sent after encoding to the server. Uses default UTF-8 encoding
client_socket.sendall(csv_data.encode())
```

### Server-side code

Below is the relevant part of the server program that is reponsible for handling the data sent by the joint tracking program and handles parsing and conversion of data for display.

``` python
# define variables for handling parsing of data
rows_received = 0
stop_flag = 0
data = b''
full_pose = []
        
# loops and handles continuous data from the client
while not stop_flag:
    # loops until all rows are parsed for one set of data from one frame
    while rows_received < self.NUM_ROWS:
        # Read up to 1024 bytes from the client and add into a temporary variable for later use
        chunk = self.request.recv(1024)
                
        # break out of loop since nothing was received
        if not chunk:
            break
        data += chunk
                
        # sets the stop flag to allow handler to exit without prematurely closing the connection
        if data.count(self.STOP_CODE):
            stop_flag = 1

        # Process each row as it is received
        while self.MARKER in data:
                    
            # split data into a row in binary string and store the rest into the same variable
            # for further iteration
            row, data = data.split(self.MARKER, 1)
                    
            # if row was empty, then we continue looking for more rows
            if not row:
                continue
                    
            # decode and split the row using the comma to separate all the values
            parsed_data = row.decode().split(',')
                    
            # convert from string to either integer or float depending on the type of data present
            # also append it to the list full_pose
            full_pose.append([int(num) if num.isdecimal() else float(num) for num in parsed_data])
                    
            # update the rows_received variable to be able to signal the outer loop whether we need to receive
            # another chunk of data or not
            rows_received += 1
                    
            # break out of the loop when all data is parsed and print the result for the frame to the console
            if rows_received == self.NUM_ROWS:
                print(full_pose)
                break
                    
        if rows_received == self.NUM_ROWS:
            break
            
    # reset all variables for next batch of data
    rows_received = 0
    data = b''
    full_pose = []
```

The server reads up to 1024 bytes of the data it receives through the client and then it goes into a loop to parse the data until all 22 rows are correctly identified and converted into a numerical format once more which can then be displayed to the console as a list. Furthermore, the entire process is repeated until the server detects the ``\0`` character which signals the end of transmission by the client. This allows the server to keep running and allow the client to disconnect and reconnect at any time.
