#include <iostream>
#include <filesystem>

#include "lane_detection_UFLDv2.h"

template<typename T>
void printPoints(const std::vector<std::vector<T>>& points)
{
    int vecIndex = 0;
    for (const auto& vec : points)
    {
        std::cout << "Vector " << vecIndex++ << ":" << std::endl;
        for (const auto& point : vec)
        {
            std::cout << point;
            
        }
        std::cout << std::endl;
    }
}

unsigned char* readData(size_t& length, const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file for reading." << std::endl;
        length = 0;
        return nullptr;
    }

    infile.seekg(0, std::ios::end);
    length = infile.tellg();
    infile.seekg(0, std::ios::beg);

    unsigned char* data = new unsigned char[length];
    infile.read(reinterpret_cast<char*>(data), length);
    infile.close();

    return data;
}

int main() 
{
    /* SET CONFIG */
    Config config;
    config.debug_level = DEBUG_LEVEL::DEBUG_ALL;
    config.engine_path = "../model/culane_res18.trt";
    config.camera_height = 143; 
    config.ue_camera_position = (cv::Mat_<int>(1, 3) << 50, 13, 143);
    config.mtx = (cv::Mat_<double>(3, 3) << 
                             1.0096978968416765e+03, 0., 9.9237751809854205e+02, 
                             0., 1.0095734048948314e+03, 7.5946186432367506e+02, 
                             0., 0., 1.);
    config.dist = (cv::Mat_<double>(1, 5) << 
                                       -3.6328291790539036e-01, 
                                       1.4643525547834677e-01, 
                                       -3.3306938293259716e-04, 
                                       7.7073209564316515e-04, 
                                       -2.7292745201340621e-02);

    /* LOAD ENGINE */
    UFLD_model engine(initializeSampleParams());
    if (config.debug_level == DEBUG_LEVEL::DEBUG_ALL) {
        std::cout << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
    }
    if (!engine.build_with_engine(config) && config.debug_level>=DEBUG_LEVEL::DEBUG_ERROR) {
        std::cerr << "build engine error" << std::endl;
        return -1;
    }

    size_t buffer_size;
    unsigned char* data;
    void* buffer;

    /* CAMERA INPUT */
    for (const auto& entry : std::filesystem::directory_iterator("../infer_input/camera/")) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") { // Saving the raw data captured by the camera as a bin file in advance
            // std::cout << entry.path().string() << std::endl;
            data = readData(buffer_size, entry.path().string()); // If input is camera, the buffer_size=1920*1536*2 (YUYV), origin type is unsigned char*
            buffer = data;
            /* infer: point2D */
            // std::vector<std::vector<cv::Point2i>> out_point2;
            // engine.infer(buffer, buffer_size, out_point2);
            // printPoints(out_point2);

            /* infer: point3D */
            std::vector<std::vector<cv::Point3i>> out_point3;
            engine.infer(buffer, buffer_size, out_point3);
            if (config.debug_level == DEBUG_LEVEL::DEBUG_ALL) {
                printPoints(out_point3);
            }
        }
    }

    /* VIDEO INPUT */
    cv::VideoCapture cap("../infer_input/video/031_20240229135140.mp4"); // Video requirements are: size 1920*1536 , uncalibrated
    if (!cap.isOpened() && config.debug_level>=DEBUG_LEVEL::DEBUG_ERROR) {
        std::cerr << "can not open the video" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while(true) {
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }

        data = frame.data;
        buffer = data;
        buffer_size = frame.total() * frame.elemSize(); // If input is video, the buffer_size=1920*1536*3 (BGR), origin type is unsigned char*

        /* infer: point2D */
        // std::vector<std::vector<cv::Point2i>> out_point2;
        // engine.infer(buffer, buffer_size, out_point2);
        // printPoints(out_point2);

        /* infer: point3D */
        std::vector<std::vector<cv::Point3i>> out_point3;
        engine.infer(buffer, buffer_size, out_point3);
        if  (config.debug_level == DEBUG_LEVEL::DEBUG_ALL) {
           printPoints(out_point3); 
        }
    }
}