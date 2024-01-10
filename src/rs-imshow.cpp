// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <vector>
#include <iostream>
#include <cmath>                // Include math library for calculations

using namespace std;

//Helper function to return magnitude of vectors
double getMagnitude(cv::Point vector);

//Function to calculate the angle using dot products
double elbowAngle(cv::Point forearm, cv::Point joint, cv::Point backarm);

int main(int argc, char * argv[]) try
{
    // Declare depth colorizer for pretty visualization of depth data
    //rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Start streaming with default recommended configuration
    pipe.start();

    using namespace cv;
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        //rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame color = data.get_color_frame();

        // Query frame size (width and height)
        const int w = color.as<rs2::video_frame>().get_width();
        const int h = color.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        //Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
        Mat colorImage(Size(w, h), CV_8UC3, (void*)color.get_data());
        Mat HSVImage;
        Mat mask_red, mask_green, mask_purple;

        Vec3i lower_red = { 160, 20, 20 };
        Vec3i upper_red = { 179, 255, 255 };

        Vec3i lower_green = {40 , 50, 50};
        Vec3i upper_green = { 80, 255, 255 };

        Vec3i lower_purple = { 150, 20, 20 };
        Vec3i upper_purple = { 159, 255, 255 };

        cvtColor(colorImage, HSVImage, COLOR_RGB2HSV);
        
        //We need to convert image from RGB to BGR as that is what openCV uses internally
        cvtColor(colorImage, colorImage, COLOR_RGB2BGR);

        inRange(HSVImage, lower_red, upper_red, mask_red);
        inRange(HSVImage, lower_green, upper_green, mask_green);
        inRange(HSVImage, lower_purple, upper_purple, mask_purple);

        //median blur on mask
        medianBlur(mask_red, mask_red, 3);
        medianBlur(mask_green, mask_green, 3);
        medianBlur(mask_purple, mask_purple, 3);

        //imshow("Purple mask", mask_purple);


        //imshow("Red Mask", mask_red);

        //find contours
        vector<vector<Point>> contoursRed, contoursGreen, contoursPurple;
        findContours(mask_red, contoursRed, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(mask_green, contoursGreen, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(mask_purple, contoursPurple, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point centerRed, centerGreen, centerPurple;

        // Get the largest red contour
        double max_area_red = -1;
        int largest_contour_index_red = -1;
        const double min_area = 500.0f;
        for (size_t i = 0; i < contoursRed.size(); ++i) {
            double area = contourArea(contoursRed[i]);
            if (area > max_area_red && area > min_area) {
                max_area_red = area;
                largest_contour_index_red = i;
            }
        }

        if (largest_contour_index_red != -1)
        {
            //drawContours(colorImage, contoursRed, largest_contour_index, Scalar(0, 0, 255), 2);
            Rect redRect = boundingRect(contoursRed.at(largest_contour_index_red));
            rectangle(colorImage, redRect, Scalar(255, 255, 255), 2);
            centerRed = { redRect.x + redRect.width / 2, redRect.y + redRect.height /2 };
            drawMarker(colorImage, centerRed, Scalar(0, 0, 255), MARKER_CROSS, 20, 3);
        }

        // Get the largest green contour
        double max_area_green = -1;
        int largest_contour_index_green = -1;
        for (size_t i = 0; i < contoursGreen.size(); ++i) {
            double area = contourArea(contoursGreen[i]);
            if (area > max_area_green && area > min_area) {
                max_area_green = area;
                largest_contour_index_green = i;
            }
        }

        if (largest_contour_index_green != -1)
        {
            //drawContours(colorImage, contoursRed, largest_contour_index, Scalar(0, 0, 255), 2);
            Rect greenRect = boundingRect(contoursGreen.at(largest_contour_index_green));
            rectangle(colorImage, greenRect, Scalar(255, 255, 255), 2);
            centerGreen = { greenRect.x + greenRect.width / 2, greenRect.y + greenRect.height / 2 };
            drawMarker(colorImage, centerGreen, Scalar(0, 255, 0), MARKER_CROSS, 20, 3);
        }

        // Get the largest purple contour
        double max_area_purple = -1;
        int largest_contour_index_purple = -1;
        for (size_t i = 0; i < contoursPurple.size(); ++i) {
            double area = contourArea(contoursPurple[i]);
            if (area > max_area_purple) {
                max_area_purple = area;
                largest_contour_index_purple = i;
            }
        }

        if (largest_contour_index_purple != -1)
        {
            //drawContours(colorImage, contoursRed, largest_contour_index, Scalar(0, 0, 255), 2);
            Rect purpleRect = boundingRect(contoursPurple.at(largest_contour_index_purple));
            rectangle(colorImage, purpleRect, Scalar(255, 255, 255), 2);
            centerPurple = { purpleRect.x + purpleRect.width / 2, purpleRect.y + purpleRect.height / 2 };
            drawMarker(colorImage, centerPurple, Scalar(255, 0, 255), MARKER_CROSS, 20, 3);
        }

        //Contour detection code
        /*Mat canny_output;
        Canny(grayscale_image, canny_output, 100, 200);

        std::vector<std::vector<Point>> contours;

        std::vector<Vec4i> hierarchy;

        findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

        const int num_contours = (int)contours.size();

        for (int i = 0; i < num_contours; i++)
        {
            double value = 255.0;

            Scalar color = Scalar(value, value, value);
            drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0);
        }*/

        int rows = HSVImage.rows;
        int cols = HSVImage.cols;
        
        Vec3b hsvValue = HSVImage.at<Vec3b>(rows / 2, cols / 2);

        drawMarker(HSVImage, Point(cols / 2, rows / 2), Scalar(255, 255, 255));

        int angle = elbowAngle(centerRed, centerPurple, centerGreen);

        // Update the window with new data
        imshow(window_name, colorImage);
        imshow("HSV Image", HSVImage);
        //cout << "HSV values: " << (int)hsvValue.val[0] << ", " << (int)hsvValue.val[1] << ", " << (int)hsvValue.val[2] << "\r";
        cout << "Elbow Angle: " << angle << '\r';
        //imshow("Contour", drawing);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

double elbowAngle(cv::Point forearm, cv::Point joint, cv::Point backarm)
{
    if (forearm.x != 0 && forearm.y != 0 && joint.x != 0 && joint.y != 0 && backarm.x != 0 && backarm.y != 0)
    {
        cv::Point forearmJoint = { forearm.x - joint.x, forearm.y - joint.y };
        cv::Point backarmJoint = { backarm.x - joint.x, backarm.y - joint.y };

        double magForearm = getMagnitude(forearmJoint), magBackarm = getMagnitude(backarmJoint);

        double theta = acos(forearmJoint.ddot(backarmJoint) / (magForearm * magBackarm));

        if (theta < 0)
        {
            theta += 2 * CV_PI;
        }

        return theta * 180.0 / CV_PI;
    }

    //Return default value when not all points are detected
    return -1;

}

double getMagnitude(cv::Point vector)
{
    double x = vector.x, y = vector.y;

    return sqrt(pow(x, 2) + pow(y, 2));

}



