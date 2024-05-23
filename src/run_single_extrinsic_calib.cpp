#include <iostream>

#include "camera.hpp"
#include "lidar.hpp"


int main(int argc, char **argv){

    std::string pcd_path = argv[1];
    std::string image_path = argv[2];
    std::string yaml_path = argv[3];

    size_t pos = yaml_path.find('/');
    std::string parent_dir = yaml_path.substr(0,pos);

    lidar lidar;
    std::vector<cv::Point3f> obj_points = lidar.pcd_keypoint(pcd_path);
    
    camera cam; 
    cam.get_cam_intrinsic(yaml_path);
    std::vector<cv::Point2f> img_points = cam.img_keypoint(image_path);

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(obj_points, img_points, cam.K, cam.D, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    if(img_points.size()>5){
        cv::solvePnP(obj_points, img_points, cam.K, cam.D, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    }
    cv::Mat rotationMatrix;

    if(success){
        cv::Rodrigues(rvec, rotationMatrix);
        
        // 输出旋转向量和平移向量
        std::cout << "rotation Matrix:\n" << rotationMatrix << std::endl;
        std::cout << "tvec:\n" << tvec << std::endl;
    }

    //外参写入yaml文件
    cv::FileStorage fw(parent_dir+"/lidar_to_camera_extrinsic.yml", cv::FileStorage::WRITE);
    fw << "Rotation_matrix" << rotationMatrix;
    fw << "Translation_vector" << tvec;

    fw.release();
    


    return 0;
}