#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>



class camera{
public:
    camera(){};
    ~camera(){};

    bool get_cam_intrinsic(const std::string &config_path);

    bool calibrate(const std::string &img_dir_path, 
                   const float &grid_size,
                   const int &corner_width, 
                   const int &corner_height,
                   bool flag);

    void undistort(const std::string &img_dir_path,
                    const cv::Mat K,
                    const cv::Mat D,
                    bool flag);

    
    std::vector<cv::Point2f> img_keypoint(const std::string &image_path);

    std::map<int, cv::Point2f> group_img_keypoint(const std::string &image_path);

    

public:
    cv::Mat K;
    cv::Mat D;  

};

std::map<int, cv::Point2f> apriltag_detect(const cv::Mat image);
