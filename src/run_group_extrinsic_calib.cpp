#include <iostream>

#include "camera.hpp"
#include "lidar.hpp"


int main(int argc, char **argv){

    std::string pcd_path = argv[1];
    std::string target_position = argv[2];

    std::string image_path = argv[3];
    std::string yaml_path = argv[4];

    size_t pos = yaml_path.find('/');
    std::string parent_dir = yaml_path.substr(0,pos);

    lidar lidar;
    std::map<int, cv::Point3f> obj_group = lidar.group_pcd_keypoint(pcd_path, target_position);

    camera cam;
    std::map<int, cv::Point2f> img_group = cam.group_img_keypoint(image_path);

    std::vector<cv::Point3f> obj_points;
    std::vector<cv::Point2f> img_points;

    for(const auto& pair : img_group) {
        img_points.push_back(pair.second);
        obj_points.push_back(obj_group[pair.first]);
        std::cout << pair.first << " -> " << pair.second << std::endl;
        std::cout << pair.first << " -> " << obj_group[pair.first] << std::endl;
    } 
    
    cam.get_cam_intrinsic(yaml_path);

    cv::Mat rvec, tvec;
    cv::Mat rotationMatrix;
    bool success = cv::solvePnP(obj_points, img_points, cam.K, cam.D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    

    if(img_points.size()>5){
        std::cout<<"keypoints more than 5, use interative solve final result!"<<std::endl;
        cv::solvePnP(obj_points, img_points, cam.K, cam.D, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    }

    cv::Rodrigues(rvec, rotationMatrix);
    
    if(success){
        // 输出旋转向量和平移向量
        // std::cout << "rotation Matrix:\n" << rotationMatrix << std::endl;
        // std::cout << "tvec:\n" << tvec << std::endl;

        cv::Mat transform_matrix = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat rotation_part = transform_matrix(cv::Rect(0, 0, 3, 3));
        rotationMatrix.copyTo(rotation_part);
        cv::Mat translation_part = transform_matrix(cv::Rect(3, 0, 1, 3));
        tvec.copyTo(translation_part);

        std::cout << "Transformation Matrix:\n" <<transform_matrix<< std::endl;

        //外参写入yaml文件
        cv::FileStorage fw(parent_dir+"/lidar_to_camera_extrinsic.yml", cv::FileStorage::WRITE);
        fw << "Rotation_matrix" << rotationMatrix;
        fw << "Translation_vector" << tvec;

        fw.release();

        
    }

    

    return 0;
    
    
}

void fit_by_icp();