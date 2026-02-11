#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "vo_features.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip> 
#include "System.h"
#include <unistd.h>
#include "CameraModels/GeometricCamera.h"
#include <sophus/se3.hpp> 

using namespace std;
using namespace cv;

#define MIN_NUM_FEAT 120 // Minimum features to track
#define MIN_AVG_MOVE 0.1 // Minium average movement of features
#define INLIERS_THRESHOLD 14 // Minimum inliers to consider a good match
#define MAX_TVEC_JUMP 0.2 // Maximum jump in translation vector
#define MAX_YAW_JUMP 0.08 // Maximum jump in yaw angle

// Convert sl::Mat to cv::Mat
cv::Mat slMat2cvMat(sl::Mat& input) {
    int cv_type = CV_8UC4;
    if (input.getDataType() == sl::MAT_TYPE::U8_C1) cv_type = CV_8UC1;
    if (input.getDataType() == sl::MAT_TYPE::U8_C3) cv_type = CV_8UC3;
    if (input.getDataType() == sl::MAT_TYPE::U8_C4) cv_type = CV_8UC4;
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

double getYawFromRotation(const cv::Mat& R) {
    return atan2(R.at<double>(1,0), R.at<double>(0,0));
}

// Detect features in the image using ORB
Point3f smoothTrajectory(const vector<Point3f>& traj, int window_size) {
    int N = traj.size();
    if (N == 0) return Point3f(0, 0, 0);
    if (N < window_size)
        return traj.back();
    float sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = N - window_size; i < N; ++i) {
        sum_x += traj[i].x;
        sum_y += traj[i].y;
        sum_z += traj[i].z;
    }
    return Point3f(sum_x / window_size, sum_y / window_size, sum_z / window_size);
}

int main() {
    // initiate ZED camera
    sl::Camera zed;
    sl::InitParameters init_params;
    
    init_params.camera_resolution = sl::RESOLUTION::VGA;
    init_params.camera_fps = 30;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.depth_mode = sl::DEPTH_MODE::PERFORMANCE;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to open ZED camera: " << sl::toString(err) << std::endl;
        return 1;
    }

    auto cam_info = zed.getCameraInformation();
    auto& calib = cam_info.camera_configuration.calibration_parameters;
    double fx = calib.left_cam.fx;
    double fy = calib.left_cam.fy;
    double cx = calib.left_cam.cx;
    double cy = calib.left_cam.cy;

    if (fx <= 1e-6 || fy <= 1e-6) {
        std::cerr << "ERROR: Invalid value. Check camera calibration parameters!" << std::endl;
        return 1;
    }
    
    // Path to ORB-SLAM3
    string vocab_file = "/path/to/your/lib-ORB-SLAM3/ORB_SLAM3/Vocabulary/ORBvoc.txt";
    string config_file = "/path/to/your/zed_config.yaml";
    
    // Initialize ORB-SLAM3 system
    ORB_SLAM3::System SLAM(vocab_file, config_file, ORB_SLAM3::System::STEREO, true);
    std::cout << "Initialized successfully!" << std::endl;

    // Camera intrinsic parameters
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(1,5,CV_64F);

    // Variables visual odometry
    sl::Mat zed_image_left, zed_image_right, depth_map;
    cv::Mat cv_frame_left, cv_frame_right, prevImage, currImage;
    vector<Point2f> prevFeatures, currFeatures;

    cv::Mat R_vo = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_vo = cv::Mat::zeros(3, 1, CV_64F);
    double theta = 0.0;
    Point3f current_position(0.0f, 0.0f, 0.0f);
    vector<Point3f> trajectory_points;

    double scale_factor = 500.0; 
    const int center_x = 400, center_y = 400;
    cv::Mat traj = cv::Mat::zeros(800, 800, CV_8UC3);

    int frame_count = 0;
    auto start_time = chrono::high_resolution_clock::now();

    if (zed.grab() != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to fetch initial frame from ZED" << std::endl;
        return 1;
    }
    
    zed.retrieveImage(zed_image_left, sl::VIEW::LEFT, sl::MEM::CPU);
    zed.retrieveImage(zed_image_right, sl::VIEW::RIGHT, sl::MEM::CPU);
    
    cv_frame_left = slMat2cvMat(zed_image_left);
    cv_frame_right = slMat2cvMat(zed_image_right);
    
    if (cv_frame_left.empty() || cv_frame_right.empty()) {
        std::cerr << "Empty frame! Failed to retrieve image from ZED." << std::endl;
        return 1;
    }

    // Convert frame to grayscale for feature detection
    if (cv_frame_left.channels() == 4) {
        cvtColor(cv_frame_left, prevImage, cv::COLOR_BGRA2GRAY);
    } else if (cv_frame_left.channels() == 3) {
        cvtColor(cv_frame_left, prevImage, cv::COLOR_BGR2GRAY);
    } else {
        prevImage = cv_frame_left.clone();
    }

    featureDetection(prevImage, prevFeatures);
    if (prevFeatures.size() == 0) {
        std::cerr << "ERROR: No features detected in the first frame! "
                  << "Try pointing the camera at a textured/bright area.." << std::endl;
        return 1;
    }

    namedWindow("Pov ", WINDOW_AUTOSIZE);
    resizeWindow("Pov ", 480, 480);

    while (true) {
        if (zed.grab() != sl::ERROR_CODE::SUCCESS) continue;
        
        zed.retrieveImage(zed_image_left, sl::VIEW::LEFT, sl::MEM::CPU);
        zed.retrieveImage(zed_image_right, sl::VIEW::RIGHT, sl::MEM::CPU);
        zed.retrieveMeasure(depth_map, sl::MEASURE::DEPTH);

        cv_frame_left = slMat2cvMat(zed_image_left);
        cv_frame_right = slMat2cvMat(zed_image_right);
        
        if (cv_frame_left.empty() || cv_frame_right.empty()) {
            std::cerr << "Empty frame! Failed to retrieve image from ZED." << std::endl;
            break;
        }

        // Convert frames for ORB-SLAM3
        cv::Mat left_gray, right_gray;
        if (cv_frame_left.channels() == 4) {
            cvtColor(cv_frame_left, left_gray, cv::COLOR_BGRA2GRAY);
            cvtColor(cv_frame_right, right_gray, cv::COLOR_BGRA2GRAY);
        } else if (cv_frame_left.channels() == 3) {
            cvtColor(cv_frame_left, left_gray, cv::COLOR_BGR2GRAY);
            cvtColor(cv_frame_right, right_gray, cv::COLOR_BGR2GRAY);
        } else {
            left_gray = cv_frame_left.clone();
            right_gray = cv_frame_right.clone();
        }

        // Detect features in the current frame
        double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // ORB-SLAM3 tracking
        Sophus::SE3f Tcw_se3 = SLAM.TrackStereo(left_gray, right_gray, timestamp);
        cv::Mat Tcw;
        bool tracking_lost = false;
        
        if (!Tcw_se3.matrix().isZero()) {
            Eigen::Matrix4f mat = Tcw_se3.matrix();
            Tcw = cv::Mat::zeros(4, 4, CV_32F);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    Tcw.at<float>(i, j) = mat(i, j);
                }
            }
        } else {
            tracking_lost = true;
        }
        
        // Update current image
        if (!Tcw.empty()) {
            cv::Mat R = Tcw.rowRange(0,3).colRange(0,3);
            cv::Mat t = Tcw.rowRange(0,3).col(3);
            
            cv::Mat Rwc = R.t();
            cv::Mat twc = -Rwc * t;
            
            current_position.x = twc.at<float>(0);
            current_position.y = twc.at<float>(1);
            current_position.z = twc.at<float>(2);
            
            theta = atan2(Rwc.at<float>(1,0), Rwc.at<float>(0,0));
            double theta_deg = theta * 180.0 / CV_PI;
            if (theta_deg < 0) theta_deg += 360.0;

            std::cout << fixed << setprecision(2)
                      << "[Pose] "
                      << " X: " << setw(8) << left << current_position.x
                      << " Y: " << setw(8) << left << current_position.y
                      << " Z: " << setw(8) << left << current_position.z
                      << "| Theta: " << setw(6) << left << theta_deg << " deg"
                      << std::endl;

            trajectory_points.push_back(current_position);
            
            if (trajectory_points.size() > 2000) {
                trajectory_points.erase(trajectory_points.begin(), trajectory_points.begin() + 500);
            }
        }

        // Visual odometry
        static float smooth_offset_x = 0.0f;
        static float smooth_offset_z = 0.0f;
        float alpha = 0.7f;

        // Smooth the trajectory
        smooth_offset_x = (1.0f - alpha) * smooth_offset_x + alpha * current_position.x;
        smooth_offset_z = (1.0f - alpha) * smooth_offset_z + alpha * current_position.z;

        // Offset for trajectory visualization
        float offset_x = smooth_offset_x;
        float offset_z = smooth_offset_z;

        // Update trajectory visualization
        float max_radius = 1.0f;
        scale_factor = 100.0;

        int adjusted_center_x = 400;
        int adjusted_center_y = 400;

        // Adjust position to center the trajectory
        traj = cv::Mat::zeros(800, 800, CV_8UC3);

        // Draw the trajectory
        line(traj, Point(adjusted_center_x - 50, adjusted_center_y), Point(adjusted_center_x + 50, adjusted_center_y), CV_RGB(100, 100, 100), 1);
        line(traj, Point(adjusted_center_x, adjusted_center_y - 50), Point(adjusted_center_x, adjusted_center_y + 50), CV_RGB(100, 100, 100), 1);
        putText(traj, "X", Point(adjusted_center_x + 55, adjusted_center_y + 5), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255), 1);
        putText(traj, "Z", Point(adjusted_center_x + 5, adjusted_center_y - 55), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255), 1);

        // Draw the trajectory points
        if (trajectory_points.size() > 1) {
            for (size_t i = 1; i < trajectory_points.size(); i++) {
                Point3f prev_pos = trajectory_points[i-1];
                Point3f curr_pos = trajectory_points[i];

                Point2i prev_screen(
                    adjusted_center_x + static_cast<int>((prev_pos.x - offset_x) * scale_factor),
                    adjusted_center_y - static_cast<int>((prev_pos.z - offset_z) * scale_factor)
                );
                Point2i curr_screen(
                    adjusted_center_x + static_cast<int>((curr_pos.x - offset_x) * scale_factor),
                    adjusted_center_y - static_cast<int>((curr_pos.z - offset_z) * scale_factor)
                );

                if (prev_screen.x >= 0 && prev_screen.x < 800 && prev_screen.y >= 0 && prev_screen.y < 800 &&
                    curr_screen.x >= 0 && curr_screen.x < 800 && curr_screen.y >= 0 && curr_screen.y < 800) {
                    Scalar color = Tcw.empty() ? CV_RGB(255, 255, 0) : CV_RGB(0, 255, 0);
                    line(traj, prev_screen, curr_screen, color, 2);
                }
            }

            // Draw the last point with an arrow
            if (!trajectory_points.empty()) {
                Point3f current_pos = trajectory_points.back();
                Point2i current_screen(
                    adjusted_center_x + static_cast<int>((current_pos.x - offset_x) * scale_factor),
                    adjusted_center_y - static_cast<int>((current_pos.z - offset_z) * scale_factor)
                );

                // Draw the current position as a circle and an arrow
                if (current_screen.x >= 0 && current_screen.x < 800 &&
                    current_screen.y >= 0 && current_screen.y < 800) {
                    circle(traj, current_screen, 5, CV_RGB(255, 0, 0), -1);

                    double arrow_len = 15;
                    double dx = arrow_len * cos(theta);
                    double dy = arrow_len * sin(theta);
                    Point2i arrow_tip(current_screen.x + static_cast<int>(dx), current_screen.y - static_cast<int>(dy));
                    arrowedLine(traj, current_screen, arrow_tip, CV_RGB(0,255,255), 2, 8, 0, 0.3);
                }
            }
        }

        // Display information window
        vector<string> info_lines;
        info_lines.push_back("X: " + to_string(current_position.x).substr(0, 6));
        info_lines.push_back("Y: " + to_string(current_position.y).substr(0, 6));
        info_lines.push_back("Z: " + to_string(current_position.z).substr(0, 6));
        
        frame_count++;
        if (frame_count % 30 == 0) {
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - start_time);
            double fps = 30000.0 / duration.count();
            info_lines.push_back("FPS: " + to_string(fps).substr(0, 4));
            start_time = current_time;
        }

        // Tracking status
        for (size_t i = 0; i < info_lines.size(); i++) {
        rectangle(traj, Point(10, 15 + i * 25), Point(400, 35 + i * 25), CV_RGB(0, 0, 0), FILLED);
        putText(traj, info_lines[i], Point(15, 30 + i * 25), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1);
        }

        imshow("Pov ", traj);

        char key = waitKey(1) & 0xFF;
        if (key == 27) break;
    }

    SLAM.Shutdown();
    // SLAM.SaveTrajectoryTUM("final_trajectory.txt");
    zed.close();
    std::cout << "Closed successfully" << std::endl;
    return 0;
}
