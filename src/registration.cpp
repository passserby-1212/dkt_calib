#include "registration.hpp"

Eigen::Matrix4f SAC(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh_src,
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh_tgt,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &sac_result){

    pcl::SampleConsensusInitialAlignment<pcl::PointXYZI, pcl::PointXYZI, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(source);
    sac_ia.setInputTarget(target);
    sac_ia.setSourceFeatures(fpfh_src);
    sac_ia.setTargetFeatures(fpfh_tgt);

    sac_ia.setMinSampleDistance(1);
    //scia.setNumberOfSamples(2);
    sac_ia.setCorrespondenceRandomness(5);

    sac_ia.align(*sac_result);

    std::cout  <<"sac has converged:"<<sac_ia.hasConverged()<<"  score: "<<sac_ia.getFitnessScore()<<endl;
    Eigen::Matrix4f sac_trans;
    sac_trans = sac_ia.getFinalTransformation();
    std::cout<<"sac_transform_matrix:\n"<<sac_trans<<endl;

    //pcl::io::savePCDFileASCII("transformed_sac.pcd",*sac_result);

    return sac_trans;
};

Eigen::Matrix4f ICP(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    Eigen::Matrix4f init_trans,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_result){
    
   
   pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
   icp.setInputSource(source);
   icp.setInputTarget(target);
   
   //icp.setMaxCorrespondenceDistance (0.2);
   icp.setMaximumIterations (50);
   //icp.setTransformationEpsilon (1e-5);
   //icp.setEuclideanFitnessEpsilon (0.2);

   icp.align(*icp_result, init_trans);

   Eigen::Matrix4f icp_trans;
   icp_trans=icp.getFinalTransformation();

   std::cout << "ICP has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
   std::cout<<"icp_transform_matrix:\n"<<icp_trans<<endl;

   return icp_trans;


}

Eigen::Matrix4f NDT(pcl::PointCloud<pcl::PointXYZI>::Ptr &source, 
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                    Eigen::Matrix4f init_trans,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_result){
    
    //初始化正态分布变换（NDT）
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    //设置依赖尺度NDT参数
    //为终止条件设置最小转换差异
    ndt.setTransformationEpsilon (0.01);
    //为More-Thuente线搜索设置最大步长
    ndt.setStepSize (0.1);
    //设置NDT网格结构的分辨率（VoxelGridCovariance）
    ndt.setResolution (1.0);
    //设置匹配迭代的最大次数
    ndt.setMaximumIterations (30);
    // 设置要配准的点云
    ndt.setInputCloud (source);
    //设置点云配准目标
    ndt.setInputTarget (target);

    ndt.align (*ndt_result, init_trans);
    Eigen::Matrix4f ndt_trans = ndt.getFinalTransformation();

    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
                << " score: " << ndt.getFitnessScore() << std::endl;
    std::cout<<"ndt_transform_matrix:\n"<<ndt_trans<<endl;

    return ndt_trans;
}

void downsample(pcl::PointCloud<pcl::PointXYZI>::Ptr &input, 
                pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                float leaf_size){

    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.setInputCloud(input);
    voxel_grid.filter(*output);

    //*output = *input;

    std::cout<<"down size *input from "<<input->size()<<"to"<<output->size()<<endl;

}


void normalestimate(pcl::PointCloud<pcl::PointXYZI>::Ptr &input,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals_output,
                    float radius){
    
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(input);
    pcl::search::KdTree< pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree< pcl::PointXYZI>());
    ne.setSearchMethod(tree);
    
    ne.setRadiusSearch(radius);
    ne.compute(*normals_output);
    std::cout<<"compute normals"<<endl;

};


void fpfh_get(pcl::PointCloud<pcl::PointXYZI>::Ptr &input,
              pcl::PointCloud<pcl::Normal>::Ptr normals_input,
              pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_o,
              float radius){

    pcl::FPFHEstimation<pcl::PointXYZI,pcl::Normal,pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(input);
    fpfh.setInputNormals(normals_input);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    fpfh.setSearchMethod(tree);

    fpfh.setRadiusSearch(radius);
    fpfh.compute(*fpfh_o);
    std::cout<<"compute fpfh"<<std::endl;

};

void visualize_pcd(pcl::PointCloud<pcl::PointXYZI>::Ptr &src, 
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &tgt,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &src_transform)
{
   //int vp_1, vp_2;
   // Create a PCLVisualizer object
   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   //viewer.createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  // viewer.createViewPort (0.5, 0, 1.0, 1.0, vp_2);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> src_h (src, 255, 255, 255);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_h (tgt, 255, 0, 0);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> final_h (src_transform, 0, 255, 0);
   //viewer.addPointCloud (src, src_h, "source cloud");
   viewer.addPointCloud (tgt, tgt_h, "tgt cloud");
   viewer.addPointCloud (src_transform, final_h, "final cloud");
   //viewer.addCoordinateSystem(1.0);
   while (!viewer.wasStopped())
   {
       viewer.spinOnce(100);
       boost::this_thread::sleep(boost::posix_time::microseconds(100000));
   }
}