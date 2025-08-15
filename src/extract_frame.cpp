#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "offline_map_recorder.hpp"
#include "tools_ros.hpp"    // Common_tools::load_obj_from_file
#include "tools_logger.hpp" // printf_program(), printf_software_version()

using PointType = pcl::PointXYZRGBA;

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.r3live> <frame_index>\n";
        return -1;
    }
    std::string filename  = argv[1];
    int         frame_idx = std::stoi(argv[2]);

    // 打印启动信息
    printf_program("R3LIVE Frame Extractor (LiDAR-frame)");
    Common_tools::printf_software_version();

    // 1. 加载离线地图
    Global_map           global_map(0);
    Offline_map_recorder recorder;
    recorder.m_global_map = &global_map;
    std::cout << "Loading offline map from: " << filename << " …" << std::endl;
    Common_tools::load_obj_from_file(&recorder, filename);

    // 检查帧索引
    size_t num_frames = recorder.m_image_pose_vec.size();
    if (frame_idx < 0 || frame_idx >= static_cast<int>(num_frames)) {
        std::cerr << "Error: frame_index out of range (0 – " 
                  << (num_frames - 1) << ")\n";
        return -1;
    }

    // 2. 取出该帧的相机位姿
    auto img = recorder.m_image_pose_vec[frame_idx];
    Eigen::Vector3d t_c2w    = img->m_pose_c2w_t;    // 相机→世界平移
    Eigen::Quaterniond q_c2w = img->m_pose_c2w_q;   // 相机→世界旋转
    Eigen::Matrix3d R_c2w    = q_c2w.toRotationMatrix();

    // 3. 相机→LiDAR 的外参（来自你的 YAML 配置）
    Eigen::Matrix3d R_lidar2cam;

    R_lidar2cam << 
        -0.00113207, -0.0158688,  0.999873,
        -0.9999999,  -0.000486594,-0.00113994,
         0.000504622,-0.999874,   -0.0158682;
         
    Eigen::Matrix3d R_cam2lidar = R_lidar2cam.transpose();
    // 若你希望使用真实平移，请取消下面一行注释并注释掉默认零平移
    // Eigen::Vector3d t_cam2lidar(0.050166, 0.0474116, -0.0312415);
    Eigen::Vector3d t_cam2lidar(0, 0, 0);

    // 4. 计算 LiDAR 在世界系下的位姿
    //    R_l2w = R_cam2lidar * R_c2w
    //    t_l2w = R_cam2lidar * t_c2w + t_cam2lidar
    Eigen::Matrix3d R_l2w = R_c2w * R_lidar2cam;
    Eigen::Vector3d t_l2w = R_c2w * t_cam2lidar + t_c2w;
    Eigen::Quaternionf q_l2w(R_l2w.cast<float>());

    std::cout << "Frame " << frame_idx << " LiDAR Pose:\n"
              << "  Translation: ["
              << t_l2w.x() << ", " << t_l2w.y() << ", " << t_l2w.z() << "]\n"
              << "  Quaternion:  ["
              << q_l2w.w() << ", " << q_l2w.x() << ", "
              << q_l2w.y() << ", " << q_l2w.z() << "]\n";

    // 5. 提取该帧的点云，转换到 LiDAR 坐标系
    auto& rgb_pts = recorder.m_pts_in_views_vec[frame_idx];
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    cloud->reserve(rgb_pts.size());

    for (const auto& pt_ptr : rgb_pts) {
        // 世界系下的点
        Eigen::Vector3d P_w = pt_ptr->get_pos();

        // 5.1 世界→相机
        Eigen::Vector3d P_c = R_c2w.transpose() * (P_w - t_c2w);

        // 5.2 相机→LiDAR
        Eigen::Vector3d P_l = R_cam2lidar * P_c + t_cam2lidar;

        // 填充 PCL 点
        PointType p;
        p.x = static_cast<float>(P_l.x());
        p.y = static_cast<float>(P_l.y());
        p.z = static_cast<float>(P_l.z());
        // 颜色顺序 B, G, R
        p.r = pt_ptr->m_rgb[2];
        p.g = pt_ptr->m_rgb[1];
        p.b = pt_ptr->m_rgb[0];
        p.a = 255;
        cloud->push_back(p);
    }

    // 6. 在 PCD header 中写入 LiDAR VIEWPOINT
    Eigen::Vector4f origin_l(
        static_cast<float>(t_l2w.x()),
        static_cast<float>(t_l2w.y()),
        static_cast<float>(t_l2w.z()),
        0.0f);
    cloud->sensor_origin_      = origin_l;
    cloud->sensor_orientation_ = q_l2w;

    // 7. 保存为二进制 PCD
    std::string out_pcd = "frame_" + std::to_string(frame_idx) + "_lidar.pcd";
    if (pcl::io::savePCDFileBinary(out_pcd, *cloud) == 0) {
        std::cout << "Saved LiDAR-frame point cloud to: " << out_pcd << std::endl;
    } else {
        std::cerr << "Failed to save PCD file." << std::endl;
        return -1;
    }

    return 0;
}

