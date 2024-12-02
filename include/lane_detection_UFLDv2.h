/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef LANE_DETECTION_UFLDV2_H
#define LANE_DETECTION_UFLDV2_H

#include "argsParser.h"
#include "buffers.h"
#include "parserOnnxConfig.h"

#include <opencv2/opencv.hpp>

enum DEBUG_LEVEL
{
    DEBUG_NONE = 0, // print nothing
    DEBUG_ERROR = 1,  // print error
    DEBUG_ALL = 2  // print all
};

typedef struct Config
{   
    std::string engine_path;
    enum DEBUG_LEVEL debug_level;  

    /* calibration */
    cv::Mat mtx;
    cv::Mat dist;
    
    /*2D --> 3D*/
    int camera_height;
    cv::Mat ue_camera_position;

    Config() : debug_level(DEBUG_LEVEL::DEBUG_NONE),
        mtx(cv::Mat::zeros(3, 3, CV_64F)), dist(cv::Mat::zeros(1, 5, CV_64F)),
        camera_height(-1), ue_camera_position(cv::Mat::zeros(1, 3, CV_32S)) { }
} Config;

samplesCommon::OnnxSampleParams initializeSampleParams();

//! \brief  The UFLD_model class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class UFLD_model
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    UFLD_model(const samplesCommon::OnnxSampleParams& params)
        : mParams(params), mInferPoints2(4), mInferPoints3(4)
    { }

    bool build_with_engine(const Config &config);
    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(const void* buffer, std::size_t buffer_size, std::vector<std::vector<cv::Point2i>> &out_point);
    bool infer(const void* buffer, std::size_t buffer_size, std::vector<std::vector<cv::Point3i>> &out_point);
    void get_infer_points_2D(std::vector<std::vector<cv::Point2i>> &points);
    void get_infer_points_3D(std::vector<std::vector<cv::Point3i>> &points);

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    nvinfer1::IRuntime* mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> mContext;

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat img);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers,cv::Mat &img_out);

private:
    bool rgb_normalize(const cv::Mat ori, cv::Mat &normlize, int crop_h);
    bool each_row_max_loc(cv::Mat ori, std::vector<int> &maxloc); 
    bool each_row_max_loc_exist(cv::Mat ori, std::vector<int> &maxloc);
    bool softmax(std::vector<float> input, std::vector<float> &re);
    std::vector<double> linspace(double start_in, double end_in, int num_in);
    void draw_line(cv::Mat &ori,const std::vector<cv::Point> lane_ps);
    void print_test(float *a, int num);
    void print_test(uchar *a, int num);
    void print_test(const std::vector<int> a,int num);
    void print_mat_size(const cv::Mat a);
    void heightToPointCloud(int u, int v, const Config &config, cv::Point3f& point);
    void point2to3(const Config &config, const std::vector<cv::Point2i>& points2, std::vector<cv::Point3i>& points3);

private:
    std::vector<std::vector<cv::Point2i>> mInferPoints2;
    std::vector<std::vector<cv::Point3i>> mInferPoints3;
    struct Config mConfig;
};

#endif