#include "camera.hpp"
#include <vector>
#include <map>

bool camera::get_cam_intrinsic(const std::string &config_path){

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file." << std::endl;
        return -1;
    }
    int device_id;
    fs["device_id"] >> device_id;
    
    cv::Mat cameraMatrix(3, 3, CV_64F);
    fs["K"] >> cameraMatrix;
    
    cv::Mat distCoeffs(1, 5, CV_64F);
    fs["D"] >> distCoeffs; 

    K = cameraMatrix;
    D = distCoeffs;
    // std::cout<<"K: "<<K<<std::endl;
    // std::cout<<"D: "<<D<<std::endl;
    return 1;
}

bool camera::calibrate(const std::string &img_dir_path, 
                        const float &grid_size,
                        const int &corner_width, 
                        const int &corner_height,
                        bool flag){

    std::vector<std::string> img_names;
    cv::glob(img_dir_path, img_names, false);

    
    cv::Mat image;
    // std::vector<cv::Mat> images;
    // for(const auto &img: img_names){
    //     image = cv::imread(img,1);
    //     images.push_back(image);
    // }

    cv::Size boardSize = cv::Size(corner_width, corner_height);

    std::vector<std::vector<cv::Point2f>> image_points; 
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<cv::Point2f> img_corners;
    std::vector<cv::Point3f> obj_corners;

    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            obj_corners.emplace_back(cv::Point3f(i * grid_size, j * grid_size, 0.0f));
        }
    }

    cv::Mat gray;
    for(const auto &img: img_names){
        std::cout<<img<<std::endl;
        image = cv::imread(img,1);
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        bool found = cv::findChessboardCorners(gray, boardSize, img_corners);
         if (found) {
            cv::cornerSubPix(gray, img_corners, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
            image_points.push_back(img_corners);
            object_points.push_back(obj_corners);

            cv::drawChessboardCorners(image, boardSize, img_corners, found);

            cv::namedWindow(img,0);
            cv::resizeWindow(img, image.cols/2,image.rows/2);
            cv::imshow(img, image);
            cv::waitKey(50);
                
        }
    }
    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms;
    if(flag){
        rms = cv::calibrateCamera(object_points, image_points, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    }else{
        rms =  cv::fisheye::calibrate(object_points, image_points, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    }
    this->K = cameraMatrix;
    this->D = distCoeffs;

    this->undistort(img_dir_path, this->K, this->D, flag);
     // 打印结果
    std::cout << "RMS error: " << rms << std::endl;
    std::cout << "Camera matrix: " << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients: " << distCoeffs << std::endl;

}

void camera::undistort(const std::string &img_dir_path,
                        const cv::Mat K,
                        const cv::Mat D,
                        bool flag){ 

    std::vector<std::string> img_names;
    cv::glob(img_dir_path, img_names, false);

    cv::Mat map1, map2;
    cv::Mat image;
    cv::Mat gray;
    for(const auto &img: img_names){
        image = cv::imread(img,1);
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::initUndistortRectifyMap(K, D, cv::Mat(), K, image.size(), CV_32FC1, map1, map2);

        cv::Mat undistorted_image;
        cv::remap(image, undistorted_image, map1, map2, cv::INTER_LINEAR);

        std::string output_path = "undistort/"+img;
        
        cv::imwrite(output_path, undistorted_image);
    }
}

std::vector<cv::Point2f> camera::img_keypoint(const std::string &image_path){
    std::vector<std::string> image_names;
    cv::glob(image_path,image_names,false);
    
    std::vector<cv::Point2f> img_points;
    for(int i=0; i<image_names.size(); i++){
        cv::Mat image = cv::imread(image_names[i],1);
        std::map<int, cv::Point2f> point = apriltag_detect(image);
        img_points.push_back(point.begin()->second);
        std::cout<<image_names[i]<<": "<<point.begin()->second<<std::endl;
    }
    return img_points;
}


std::map<int, cv::Point2f> camera::group_img_keypoint(const std::string &image_path){
    
    cv::Mat image = cv::imread(image_path, 1);
    std::map<int, cv::Point2f> group = apriltag_detect(image);

    return group;  
}

std::map<int, cv::Point2f> apriltag_detect(const cv::Mat image){
    // 创建AprilTag检测器
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    // 设定检测参数
    td->quad_decimate = 1.0;  // 减少图像采样率以提高速度
    td->quad_sigma = 0.0;     // 推荐值为0.0，用于过滤边缘噪声
    td->nthreads = 4;         // 使用多线程检测

    // 转换图像格式为灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 创建图像包装类
    image_u8_t im = { .width = gray.cols, .height = gray.rows, .stride = gray.cols, .buf = gray.data };

    // 检测AprilTag
    zarray_t *detections = apriltag_detector_detect(td, &im);
    cv::Mat H;

    // 遍历检测到的AprilTag
    std::map<int, cv::Point2f> group;
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        
        cv::Point2f point(det->c[0], det->c[1]);
        int id = det->id;
       
        group.insert(std::make_pair(id, point));
        
        std::cout<<"ID = "<< det->id <<", Center = ("<<det->c[0]<<", "<<det->c[1]<<")"<< std::endl;
        // std::cout<<"corner: ("<<det->p[0][0]<<','<<det->p[0][1]<<')'<<std::endl;
        // std::cout<<"corner: ("<<det->p[1][0]<<','<<det->p[1][1]<<')'<<std::endl;
        // std::cout<<"corner: ("<<det->p[2][0]<<','<<det->p[2][1]<<')'<<std::endl;
        // std::cout<<"corner: ("<<det->p[3][0]<<','<<det->p[3][1]<<')'<<std::endl;
        
        //在图像上绘制AprilTag的边界框
        // cv::line(image, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 255, 0), 2);
        // cv::line(image, cv::Point(det->p[1][0], det->p[1][1]), cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0, 255, 0), 2);
        // cv::line(image, cv::Point(det->p[2][0], det->p[2][1]), cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 255, 0), 2);
        // cv::line(image, cv::Point(det->p[3][0], det->p[3][1]), cv::Point(det->p[0][0], det->p[0][1]), cv::Scalar(0, 255, 0), 2);
        // //在图像上绘制Apriltag的角点
        // cv::circle(image, cv::Point(det->p[0][0], det->p[0][1]), 5, cv::Scalar(0, 0, 255), 2);
        // cv::circle(image, cv::Point(det->p[1][0], det->p[1][1]), 5, cv::Scalar(0, 0, 255), 2);
        // cv::circle(image, cv::Point(det->p[2][0], det->p[2][1]), 5, cv::Scalar(0, 0, 255), 2);
        // cv::circle(image, cv::Point(det->p[3][0], det->p[3][1]), 5, cv::Scalar(0, 0, 255), 2);
        // //在图像上绘制Apriltag的中心点
        // cv::circle(image, cv::Point(det->c[0], det->c[1]), 5, cv::Scalar(0, 0, 255), -1);
        // // 在图像上显示AprilTag的ID
        // cv::putText(image, std::to_string(det->id), cv::Point(det->c[0]-100, det->c[1]+100), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }

    // 释放内存
    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    // 显示标记后的图像
    // cv::imwrite(std::to_string(id+1) + ".jpg",image);
    // cv::namedWindow("AprilTag Detection", 0);
    // cv::resizeWindow("AprilTag Detection", 1080, 1920);
    // cv::imshow("AprilTag Detection", image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    
    return group;
}

