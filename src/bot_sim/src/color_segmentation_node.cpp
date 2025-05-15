#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <filesystem>
#include <iostream>

#include "bot_sim/utils.h"
#include "bot_sim/yolov8Predictor.h"

class YoloObjectTrackingNode : public rclcpp::Node {
public:
    YoloObjectTrackingNode()
        : Node("yolo_object_tracking_node"),
          modelPath_("/home/ichan/ros2_sim/src/bot_sim/models/zed_obs.onnx"),
          classNamesPath_("/home/ichan/ros2_sim/src/bot_sim/models/coco.names"),
          confThreshold_(0.5f), iouThreshold_(0.5f), maskThreshold_(0.6f), isGPU_(true),
          initialized_(false) {

        RCLCPP_INFO(this->get_logger(), "YoloObjectTrackingNode started (Webcam)");

        classNames_ = utils::loadNames(classNamesPath_);
        if (classNames_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Class name file is empty.");
            rclcpp::shutdown();
            return;
        }

        if (!std::filesystem::exists(modelPath_)) {
            RCLCPP_ERROR(this->get_logger(), "Model file does not exist.");
            rclcpp::shutdown();
            return;
        }

        try {
            predictor_ = std::make_unique<YOLOPredictor>(modelPath_, isGPU_, confThreshold_, iouThreshold_, maskThreshold_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "YOLO init error: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        publisher_ = this->create_publisher<geometry_msgs::msg::Point>("detected_object", 10);

        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open webcam");
            rclcpp::shutdown();
            return;
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&YoloObjectTrackingNode::timer_callback, this));

        initialized_ = true;
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string modelPath_, classNamesPath_;
    float confThreshold_, iouThreshold_, maskThreshold_;
    bool isGPU_;
    bool initialized_;

    std::vector<std::string> classNames_;
    std::unique_ptr<YOLOPredictor> predictor_;
    cv::VideoCapture cap_;

    void timer_callback() {
        if (!initialized_) return;

        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "Failed to read frame from webcam");
            return;
        }

        auto results = predictor_->predict(frame);
        utils::visualizeDetection(frame, results, classNames_);

        geometry_msgs::msg::Point msg_point;
        bool detected = false;

        for (const auto& res : results) {
            int cx = res.box.x + res.box.width / 2;
            int cy = res.box.y + res.box.height / 2;
            if (res.classId >= 0 && res.classId < static_cast<int>(classNames_.size()) &&
                classNames_[res.classId] == "Ball") {
                msg_point.x = cx;
                msg_point.y = cy;
                msg_point.z = 0;
                publisher_->publish(msg_point);
                detected = true;
                RCLCPP_INFO(this->get_logger(), "Detected Ball at (%.1f, %.1f)", msg_point.x, msg_point.y);
                break;
            }
        }

        if (!detected) {
            msg_point.x = msg_point.y = msg_point.z = -1;
            publisher_->publish(msg_point);
        }

        cv::imshow("YOLO + Webcam", frame);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloObjectTrackingNode>());
    rclcpp::shutdown();
    return 0;
}
