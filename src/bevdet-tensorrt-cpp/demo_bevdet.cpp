#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>

#include <yaml-cpp/yaml.h>
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



using std::chrono::duration;
using std::chrono::high_resolution_clock;

void Getinfo(void) {
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);
    }
    printf("\n");
}


void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel=false) {
  std::ofstream out_file;
  out_file.open(file_name, std::ios::out);
  if (out_file.is_open()) {
    for (const auto &box : boxes) {
      out_file << box.x << " ";
      out_file << box.y << " ";
      out_file << box.z << " ";
      out_file << box.l << " ";
      out_file << box.w << " ";
      out_file << box.h << " ";
      out_file << box.r << " ";
      if(with_vel){
        out_file << box.vx << " ";
        out_file << box.vy << " ";
      }
      out_file << box.score << " ";
      out_file << box.label << "\n";
    }
  }
  out_file.close();
  return;
};


void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        std::vector<Box> &lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans){
    for(size_t i = 0; i < ego_boxes.size(); i++){
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= lidar2ego_trans.translation();
        center = lidar2ego_rot.inverse().matrix() * center;
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        lidar_boxes.push_back(b);
    }
}


void TestNuscenes(YAML::Node &config){
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string data_info_path = config["dataset_info"].as<std::string>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    std::string output_dir = config["OutputDir"].as<std::string>();
    std::vector<std::string> cams_name = config["cams"].as<std::vector<std::string>>();

    DataLoader nuscenes(img_N, img_h, img_w, data_info_path, cams_name);
    BEVDet bevdet(model_config, img_N, nuscenes.get_cams_intrin(), 
            nuscenes.get_cams2ego_rot(), nuscenes.get_cams2ego_trans(), imgstage_file,
            bevstage_file);
    std::vector<Box> ego_boxes;
    double sum_time = 0;
    int  cnt = 0;
    for(int i = 0; i < nuscenes.size(); i++){
        ego_boxes.clear();
        float time = 0.f;
        bevdet.DoInfer(nuscenes.data(i), ego_boxes, time, i);
        if(i != 0){
            sum_time += time;
            cnt++;
        }
        Boxes2Txt(ego_boxes, output_dir + "/bevdet_egoboxes_" + std::to_string(i) + ".txt", true);
    }
    printf("Infer mean cost time : %.5lf ms\n", sum_time / cnt);
}

void TestSample(YAML::Node &config){
    printf("==3==\n");
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    YAML::Node camconfig = YAML::LoadFile(config["CamConfig"].as<std::string>()); 
    std::string output_lidarbox = config["OutputLidarBox"].as<std::string>();
    YAML::Node sample = config["sample"];
    std::string output_dir = config["OutputDir"].as<std::string>();
    printf("==4==\n");
    std::vector<std::string> imgs_file;
    std::vector<std::string> imgs_name;
    for(auto file : sample){
        std::cout << file.second.as<std::string>() << std::endl;
        imgs_file.push_back(file.second.as<std::string>());
        imgs_name.push_back(file.first.as<std::string>()); 
    }
    // // 遍历 imgs_file 向量
    // for (const std::string& file : imgs_file) {
    //     std::cout << file << std::endl;
    // }

    // // 遍历 imgs_name 向量
    // for (const std::string& name : imgs_name) {
    //     std::cout << name << std::endl;
    // }
    // printf("imgs_file.size = %d \n", imgs_file.size());
    camsData sampleData;
    sampleData.param = camParams(camconfig, img_N, imgs_name);

    BEVDet bevdet(model_config, img_N, sampleData.param.cams_intrin, 
                sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans, 
                                                    imgstage_file, bevstage_file);
    std::vector<std::vector<char>> imgs_data;
    read_sample(imgs_file, imgs_data);

    std::cout << "imgs_data = " << imgs_data.size() << std::endl;
    int count = 0;
    for (const auto& inner_vec : imgs_data) {
        for (char c : inner_vec) {
            count++;
        }
        std::cout << "==========count===========" << count;
        count = 0;
        std::cout << std::endl; // 在内部向量遍历结束后换行
    }
    uchar* imgs_dev = nullptr;
    std::cout << "size2 = " << img_N * 3 * img_w * img_h * sizeof(uchar) << std::endl;
    CHECK_CUDA(cudaMalloc((void**)&imgs_dev, img_N * 3 * img_w * img_h * sizeof(uchar)));
    decode_cpu(imgs_data, imgs_dev, img_w, img_h);
    sampleData.imgs_dev = imgs_dev;

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
    float time = 0.f;
    bevdet.DoInfer(sampleData, ego_boxes, time);
    std::vector<Box> lidar_boxes;
    Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData.param.lidar2ego_rot, 
                                            sampleData.param.lidar2ego_trans);
    Boxes2Txt(lidar_boxes, output_lidarbox, false);
    ego_boxes.clear();
    // bevdet.DoInfer(sampleData, ego_boxes, time); // only for inference time
    

}

cv::Mat zed_image;
YAML::Node config;
void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }    
    // num++;
    cv_ptr->image.copyTo(zed_image);

    cv::imshow("zed_image", zed_image);
    cv::waitKey(1);

    printf("==2==\n");
    TestSample(config);
  }


int main(int argc, char **argv){
    Getinfo();
    // if(argc < 2){
    //     printf("Need a configure yaml! Exit!\n");
    //     return 0;
    // }
    std::string config_file("./src/bevdet/src/bevdet-tensorrt-cpp/configure.yaml");
    config = YAML::LoadFile(config_file);
    printf("Successful load config : %s!\n", config_file.c_str());
    
    // 初始化 ROS 节点
    ros::init(argc, argv, "image_subscriber");

    // 创建 ROS 句柄
    ros::NodeHandle nh;
    image_transport::Subscriber image_sub_;
    image_transport::ImageTransport it(nh);
    // 创建订阅器，订阅图像消息
    image_sub_ = it.subscribe("/zed2i/zed_node/rgb/image_rect_color", 1, &imageCb);
    // 进入 ROS 循环
    ros::spin();
    
    


    return 0;
}