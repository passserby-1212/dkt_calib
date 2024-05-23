#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>   
#include <pcl/point_types.h> 
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/project_inliers.h>


#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

class lidar{
public:
    lidar(){};
    ~lidar(){};
    //单标靶
    std::vector<cv::Point3f> pcd_keypoint(const std::string &pcd_path);
    //多标靶
    std::map<int, cv::Point3f> group_pcd_keypoint(const std::string &pcd_path, const std::string &target_position);

    std::map<int, cv::Point3f> get_target_position(const std::string &target_position);

};

Eigen::Matrix4d solve_transform_matrix(Eigen::Vector3d v0, Eigen::Vector3d v1);

bool is_squarePlane(pcl::PointCloud<pcl::PointXYZI>::Ptr &input);

pcl::PointCloud<pcl::PointXYZI>::Ptr multiframes(std::vector<std::string> &pcd_names);

std::vector<cv::Point3f> cluster(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, int frames);

void passFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                std::string field,
                int intensity_min,
                int intensity_max);

void statisticalFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                       pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                       int MeanK, float std);

void plane_fit(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
               pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
               float distance_threshold);

cv::Point3f get_center(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);