#pragma once

#include <iostream>

#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>


Eigen::Matrix4f SAC(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh_src,
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh_tgt,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &sac_result);

Eigen::Matrix4f ICP(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    Eigen::Matrix4f init_trans,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_result);

Eigen::Matrix4f NDT(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    Eigen::Matrix4f init_trans,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_result);


void downsample(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                float leaf_size);

void normalestimate(pcl::PointCloud<pcl::PointXYZI>::Ptr &input,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals_output,
                    float radius);

void fpfh_get(pcl::PointCloud<pcl::PointXYZI>::Ptr &input,
          pcl::PointCloud<pcl::Normal>::Ptr normals_input,
          pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh,
          float radius);


void visualize_pcd(pcl::PointCloud<pcl::PointXYZI>::Ptr &src, 
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &tgt,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &src_transform);