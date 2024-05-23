#include "lidar.hpp"


std::vector<cv::Point3f> lidar::pcd_keypoint(const std::string &pcd_path){
    std::vector<std::string> pcd_names;
    cv::glob(pcd_path, pcd_names, false);

    std::vector<cv::Point3f> obj_points;
    for(int i=0; i<pcd_names.size(); i++){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        

        pcl::PCDReader reader;
        reader.read(pcd_names[i], *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_i(new pcl::PointCloud<pcl::PointXYZI>);
        passFilter(cloud, cloud_i, "intensity",180, 255);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZI>);
        plane_fit(cloud_i, cloud_p, 0.01f);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_final(new pcl::PointCloud<pcl::PointXYZI>);
        statisticalFilter(cloud_p, cloud_final, 50, 3.0f);

        //存储提取的pcd文件
        //pcl::io::savePCDFile(std::to_string(i+1)+".pcd", *cloud_final);

        cv::Point3f center = get_center(cloud_final);
        obj_points.push_back(center);
        std::cout<<pcd_names[i]<<": "<<center<<std::endl;

    }
    return obj_points;

}

std::map<int, cv::Point3f> lidar::group_pcd_keypoint(const std::string &pcd_path,
                                                     const std::string &target_position){
    std::vector<std::string> pcd_names;
    cv::glob(pcd_path, pcd_names, false);
    int frames = pcd_names.size();

    pcl::PointCloud<pcl::PointXYZI>::Ptr group = multiframes(pcd_names);

    //筛选出标靶点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr group_i(new pcl::PointCloud<pcl::PointXYZI>);
    passFilter(group, group_i, "intensity", 170, 255);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr group_f(new pcl::PointCloud<pcl::PointXYZI>);
    statisticalFilter(group_i, group_f, 50, 1.0f);
    //pcl::io::savePCDFile("filtered.pcd", *group_f);

    //获取标靶中心点
    std::vector<cv::Point3f> obj_points = cluster(group_f, frames);
    
    //标靶点云坐标和真值坐标进行匹配
    pcl::PointCloud<pcl::PointXYZ>::Ptr acutal_position(new pcl::PointCloud<pcl::PointXYZ>);
    cv2pcl(obj_points, acutal_position);

    int count = obj_points.size();
    std::map<int, cv::Point3f> truth_points = get_target_position(target_position);

    std::vector<cv::Point3f> tgt_position;
    for(const auto& pair : truth_points){
        tgt_position.push_back(truth_points[pair.first]);
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr truth_position(new pcl::PointCloud<pcl::PointXYZ>);
    cv2pcl(tgt_position, truth_position);

    Eigen::Matrix4f trans = coordinates_fit(acutal_position, truth_position);

    std::map<int, cv::Point3f> actual_points;
    for(const auto& pair : truth_points) {
        float bias = 100.0f;
        for(const auto pt: obj_points){
            Eigen::Vector4f point(pt.x, pt.y, pt.z ,1.0f);
            Eigen::Vector4f point_t = trans * point;

            cv::Point3f point_res(point_t.x()/point_t.w(),point_t.y()/point_t.w(),point_t.z()/point_t.w());
            cv::Point3f diff = pair.second - point_res;
            if(cv::norm(diff) < bias){
                actual_points[pair.first] = pt;
                bias = cv::norm(diff);

            };  
        }   
    }

    return actual_points;  

}

std::map<int, cv::Point3f> lidar::get_target_position(const std::string &target_position){

    std::map<int, cv::Point3f> obj_truth_value;

    cv::FileStorage fs(target_position, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file." << std::endl;
    }
    cv::FileNode stationNode = fs["station_xxx"];
    for (cv::FileNodeIterator it = stationNode.begin(); it != stationNode.end(); ++it) {
        cv::FileNode target = *it;
        std::string plane = target.name();
        
        size_t pos = plane.find('_');
        
        cv::Point3f coordinates;
        target >> coordinates;
        int id = std::stoi(plane.substr(pos+1,plane.size()-1));

        obj_truth_value.insert({id, coordinates});

        //std::cout<<id<<": "<< coordinates<<std::endl;
        
    }
    return obj_truth_value;
}

Eigen::Matrix4d solve_transform_matrix(Eigen::Vector3d v0, Eigen::Vector3d v1){
    // 计算旋转轴
    Eigen::Vector3d r = v0.cross(v1).normalized();
    // 计算旋转角度
    double cos_theta = v0.dot(v1) / (v0.norm() * v1.norm());
    double theta = std::acos(cos_theta);

    Eigen::Matrix3d rotation_matrix; 
    rotation_matrix = Eigen::AngleAxisd(theta, r);

    Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity();
    transform_matrix.block<3, 3>(0, 0) = rotation_matrix;

    return transform_matrix;
}

bool is_squarePlane(pcl::PointCloud<pcl::PointXYZI>::Ptr &input){
    // 创建点云平面拟合对象
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(input);
    seg.segment(*inliers, *coefficients);
    
    Eigen::Vector3d v0(coefficients->values[0], coefficients->values[1], coefficients->values[2]); // 初始方向
    Eigen::Vector3d v1(1.0, 0.0, 0.0); // 最终方向
    Eigen::Matrix4d T_matrix = solve_transform_matrix(v0, v1);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud (*input, *cloud_projected, T_matrix);
    // 计算投影后点云的边界框
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud_projected, min_pt, max_pt);
    
    // 计算边界框的长宽比
    float length = max_pt[1] - min_pt[1];
    float width = max_pt[2] - min_pt[2];
    //std::cout<<"length: width-->"<<length<<": "<<width<<std::endl;
    float aspect_ratio = length / width;
    //std::cout<<"length/width->ratio: "<<aspect_ratio<<std::endl;

    // 判断是否是正方形
    float tolerance = 0.15; // 容忍的长宽比差异
    if (std::abs(aspect_ratio - 1.0) < tolerance)
    {
        return 1;
    }
    
    return 0;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr multiframes(std::vector<std::string> &pcd_names){
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PCDReader reader;

    for(auto pcd:pcd_names){
        reader.read(pcd, *temp);
        *cloud += *temp;
        temp->clear();
    }
    return cloud;
}

std::vector<cv::Point3f> cluster(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, int frames){
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud (input);
    pcl::PCDWriter writer;

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;   //欧式聚类对象
    ec.setClusterTolerance (0.1f);                     // 设置近邻搜索的搜索半径为10cm
    ec.setMinClusterSize (100);                 //设置一个聚类需要的最少的点数目
    ec.setMaxClusterSize (1000*frames);               //设置一个聚类需要的最大点数目
    ec.setSearchMethod (tree);                    //设置点云的搜索机制
    ec.setInputCloud (input);
    ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中

    std::cout<<"clusters numbers: "<<cluster_indices.size()<<std::endl;

    std::vector<cv::Point3f> center_points;
    int j = 1;
    for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
        cloud_cluster->points.push_back(input->points[*pit]); 
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        }
        //std::cout<<"cluster size: "<<cloud_cluster->width<<std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr single(new pcl::PointCloud<pcl::PointXYZI>);

        statisticalFilter(cloud_cluster, single, 50, 1.0f);
        pcl::PointCloud<pcl::PointXYZI>::Ptr single_f(new pcl::PointCloud<pcl::PointXYZI>);
        plane_fit(single, single_f, 0.01f);

        bool plane = is_squarePlane(single_f);
        
        if(plane){
            //pcl::io::savePCDFile("plane"+ std::to_string(j)+".pcd", *single_f);
            cv::Point3f center = get_center(single_f);

            std::cout<<"plane_"+std::to_string(j)+": "<<center<<std::endl;

            center_points.push_back(center);
            j++;
            

            //存储筛选的标靶点云
            // std::stringstream ss;
            // ss << "./cloud_plane_" << j << ".pcd";
            // writer.write<pcl::PointXYZI> (ss.str (), *single, false);
        }
    }
    return center_points;
}

void passFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                std::string field,
                int intensity_min,
                int intensity_max){

    pcl::PassThrough<pcl::PointXYZI> pass;
    
    //根据intensity筛选特征点云
    pass.setInputCloud (input);            
    pass.setFilterFieldName (field);
    pass.setFilterLimits (intensity_min, intensity_max);

    pass.filter (*output);
}

void statisticalFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                       pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                       int MeanK, float std){

    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud (input);                           
    sor.setMeanK (MeanK);                               
    sor.setStddevMulThresh (std);

    sor.filter (*output); 
}

void plane_fit(pcl::PointCloud<pcl::PointXYZI>::Ptr &input,
               pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
               float distance_threshold){
    // 使用 RANSAC 算法拟合平面
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(distance_threshold);
    seg.setInputCloud(input);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
        return;
    }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud (input);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*output);
}

cv::Point3f get_center(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud){
    // 初始化最大和最小值
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();

  // 遍历点云数据
  for (const auto& point : *cloud)
  {
    // 更新最大和最小值
    if (point.x < min_x) min_x = point.x;
    if (point.y < min_y) min_y = point.y;
    if (point.z < min_z) min_z = point.z;
    if (point.x > max_x) max_x = point.x;
    if (point.y > max_y) max_y = point.y;
    if (point.z > max_z) max_z = point.z;
  }
   
  float center_x = (min_x+max_x)/2;
  float center_y = (min_y+max_y)/2;
  float center_z = (min_z+max_z)/2;

  cv::Point3f point_c(center_x,center_y,center_z);
  
  return point_c;
  
}

void lidar::cv2pcl(const std::vector<cv::Point3f> &cv_pt, 
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_pt){
    for (const auto& point : cv_pt)
    {
        pcl::PointXYZ pclPoint;
        pclPoint.x = point.x;
        pclPoint.y = point.y;
        pclPoint.z = point.z;
        pcl_pt->push_back(pclPoint);
    }

}

void lidar::pcl2cv(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_pt, 
                                    std::vector<cv::Point3f> &cv_pt){
    for (const auto& point : pcl_pt->points)
    {
        cv::Point3f cvPoint(point.x, point.y, point.z);
        cv_pt.push_back(cvPoint);
    }

}

Eigen::Matrix4f lidar::coordinates_fit(const pcl::PointCloud<pcl::PointXYZ>::Ptr &src,
                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &tgt){
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
   
   //icp.setMaxCorrespondenceDistance (0.2);
   icp.setMaximumIterations (50);
   //icp.setTransformationEpsilon (1e-5);
   //icp.setEuclideanFitnessEpsilon (0.2);

   icp.align(*icp_result);

   Eigen::Matrix4f icp_trans = icp.getFinalTransformation();
   std::cout<<"icp_trans:\n"<<icp_trans<<std::endl;
   return icp_trans;
   

}