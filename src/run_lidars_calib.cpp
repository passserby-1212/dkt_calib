#include <iostream>

#include "lidar.hpp"
#include "registration.hpp"

int main(int argc, char **argv){

    std::string target = argv[1];
    std::string source = argv[2];

    pcl::PointCloud<pcl::PointXYZI>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(target, *tgt);
    pcl::io::loadPCDFile(source, *src);
    //去除NAN点
    std::vector<int> indices_src, indices_tgt;
    pcl::removeNaNFromPointCloud(*tgt, *tgt, indices_tgt); 
    pcl::removeNaNFromPointCloud(*src, *src, indices_src);

    /*筛选特征*/
    pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_f1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_f2(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PointCloud<pcl::PointXYZI>::Ptr src_f1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr src_f2(new pcl::PointCloud<pcl::PointXYZI>);
    //center-left-y[(0,5),(-4,1)], center-right-y[(-7,-2),(1,6)]
    passFilter(tgt, tgt_f1, "x", 6, 20);
    passFilter(tgt_f1, tgt_f2, "y", -7,-2);

    passFilter(src, src_f1, "x", 6, 20);
    passFilter(src_f1, src_f2, "y", 1,6);

    
    pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_down(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr src_down(new pcl::PointCloud<pcl::PointXYZI>);
    std::cout<<"target downsample"<<std::endl;
    downsample(tgt_f2, tgt_down, 0.1f);
    std::cout<<"source downsample"<<std::endl;
    *src_down = *src_f2;
    //downsample(src_f2, src_down, 0.1f);


    //计算fpfh特征
    pcl::PointCloud<pcl::Normal>::Ptr tgt_norm(new pcl::PointCloud< pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr src_norm(new pcl::PointCloud< pcl::Normal>);
    float radius_tgt = 1.0f;
    float radius_src = 1.0f;
    normalestimate(tgt_down, tgt_norm, radius_tgt);
    normalestimate(src_down, src_norm, radius_src);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_src(new pcl::PointCloud<pcl::FPFHSignature33>());
    float fpfh_radius_tgt = 2.0f;
    float fpfh_radius_src = 2.0f;
    fpfh_get(tgt_down, tgt_norm, fpfh_tgt, fpfh_radius_tgt);
    fpfh_get(src_down, src_norm, fpfh_src, fpfh_radius_src);

    //点云配准
    pcl::PointCloud<pcl::PointXYZI>::Ptr src_sac(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f init_guess = SAC(src_down, tgt_down, fpfh_src, fpfh_tgt, src_sac);
    //Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr src_ndt(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f ndt_trans = NDT(src_down, tgt_down, init_guess, src_ndt);
    pcl::transformPointCloud(*src, *src_ndt, ndt_trans);

    pcl::PointCloud<pcl::PointXYZI>::Ptr src_icp(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f icp_trans = ICP(src_down, tgt_down, ndt_trans, src_icp);
    pcl::transformPointCloud(*src, *src_icp, icp_trans);

    //可视化
    visualize_pcd(src_f2, tgt, src_ndt);
    //visualize_pcd(src_f2, tgt_down, src_icp);



    return 0;
}