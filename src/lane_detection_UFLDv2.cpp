#include "lane_detection_UFLDv2.h"

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "logger.h"

#include "yuv2rgb.cuh"

static cv::Rect roi(0, 228, 1920, 1080);

bool UFLD_model::build_with_engine(const Config &config)
{ 
    this->mConfig = config;
    std::string loadEngine = mConfig.engine_path;

    if (loadEngine.size() > 0)
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            // std::cout << "size:" <<size <<std::endl;
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        mRuntime = nvinfer1::createInferRuntime(sample::gLogger);
        mEngine = SampleUniquePtr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());
        
        // mRuntime->destroy();

        mContext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext)
        {
            return false;
        }
        // clock_t end_buff=clock();
        // std::cout << "buff allocation and context checking time: " << double(end_buff-start_buff)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
        if (!mEngine)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

static void CudaUndistort(cv::Mat &bgr_mat, cv::Mat &undist_bgr_mat, cv::Mat mtx, cv::Mat dist)
{
	cv::cuda::GpuMat src(bgr_mat);
	cv::cuda::GpuMat distortion(src.size(),src.type());

	cv::Size imageSize = src.size();

	cv::Mat map1, map2;
	initUndistortRectifyMap(
		mtx, dist, cv::Mat(),
		mtx, imageSize,
		CV_32FC1, map1, map2);

	cv::cuda::GpuMat m_mapx;
	cv::cuda::GpuMat m_mapy;
	m_mapx = cv::cuda::GpuMat(map1);
	m_mapy = cv::cuda::GpuMat(map2);

	cv::cuda::remap(src, distortion, m_mapx, m_mapy, cv::INTER_LINEAR);
	distortion.download(undist_bgr_mat);
}

bool UFLD_model::infer(const void* img_buffer_, std::size_t buffer_size, std::vector<std::vector<cv::Point2i>> &out_points)
{
    
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    // gpuConvertYUYVtoBGR((unsigned char *)ctx.g_buff[v4l2_buf.index].start, ctx.cuda_out_buffer, ctx.cam_w, ctx.cam_h);        
    mInferPoints2.clear();
    
    void* img_buffer = const_cast<void*>(img_buffer_);
    unsigned char* cuda_out_buffer = nullptr;
    if (buffer_size == 1920*1536*3) { //video input, BGR
        cuda_out_buffer = static_cast<unsigned char*>(img_buffer);  
    }
    else if (buffer_size == 1920*1536*2) { //camera input, YUYV
        std::cout << __FILE__ << ":" << __LINE__ << std::endl;

        cuda_out_buffer = (unsigned char*) malloc(1920*1536*3);
        gpuConvertYUYVtoBGR(static_cast<unsigned char*>(img_buffer), cuda_out_buffer, 1920, 1536);   
    }
    else {
        if (this->mConfig.debug_level >= DEBUG_LEVEL::DEBUG_ERROR) {
            std::cerr << __FILE__ << ":" <<__LINE__ << ": " << std::endl << "error void* buffer " << std::endl;
        }
        return false;
    }
          
    cv::Mat img;
    cv::Mat bgr_mat(1536, 1920, CV_8UC3, cuda_out_buffer);
    cudaDeviceSynchronize();
    
    /* Undistort: CUDA boost */
    cv::Mat undist_bgr_mat;
    CudaUndistort(bgr_mat, undist_bgr_mat, this->mConfig.mtx, this->mConfig.dist);
    /* Undistort */ 
    // cv::undistort(bgr_mat, undist_bgr_mat, mtx, dist);

    if (buffer_size == 1920*1536*2 && cuda_out_buffer != nullptr) {
        free(cuda_out_buffer);
        cuda_out_buffer = nullptr;
    }
    
    /* Crop to 1920x1080 */ 
    cv::Mat infer_mat = undist_bgr_mat(roi);

    /* Resize to 1280x720 */ 
    cv::resize(infer_mat, infer_mat, cv::Size(1280, 720)); 

    // engine.mInferPoints2.clear();
    // engine.mInferPoints3.clear();

    // img_out = img.clone();
    
    samplesCommon::BufferManager buffers(mEngine);
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    // clock_t start=clock();

    if (!processInput(buffers, infer_mat))
    {
        return false;
    }
    // clock_t end=clock();
    // std::cout << "preprocessing time: " << double(end-start)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    // Memcpy from host input buffers to device input buffers
    // clock_t start_ref=clock();

    buffers.copyInputToDevice();
    // mContext->setOptimizationProfileAsync(0);
    bool status = mContext->executeV2(buffers.getDeviceBindings().data());

    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // clock_t end_ref=clock();
    // std::cout << "infer time(include mem cp): " << double(end_ref-start_ref)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    
    // clock_t start_post=clock();
    // Verify results
    if (!verifyOutput(buffers, infer_mat))
    {
        return false;
    }
    // clock_t end_post=clock();

    if (this->mConfig.debug_level == DEBUG_LEVEL::DEBUG_ALL) {
        cv::imshow("camera", infer_mat);
        cv::waitKey(1);
    }


    get_infer_points_2D(out_points);
    
    return true;
    // std::cout << "post processing time: " << double(end_post-start_post)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;    

}

bool UFLD_model::infer(const void* img_buffer_, std::size_t buffer_size, std::vector<std::vector<cv::Point3i>> &out_points)
{
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    mInferPoints2.clear();  
    mInferPoints3.clear();  

    void* img_buffer = const_cast<void*>(img_buffer_);
    // std::cout << "sizeof(void*)= " << sizeof(void*) << std::endl;
    unsigned char* cuda_out_buffer = nullptr;

    if (buffer_size == 1920*1536*3) { // video input, BGR
        cuda_out_buffer = (unsigned char*)img_buffer;  
    }
    else if (buffer_size == 1920*1536*2) { //camera input, YUYV
        cuda_out_buffer = (unsigned char*) malloc(1920*1536*3);
        gpuConvertYUYVtoBGR(static_cast<unsigned char*>(img_buffer), cuda_out_buffer, 1920, 1536);   
    }
    else {
        if (this->mConfig.debug_level >= DEBUG_LEVEL::DEBUG_ERROR) {
            std::cerr << __FILE__ << ":" <<__LINE__ << ": " << std::endl << "Invalid void* buffer, please input 1920*1536*3 or 1920*1536*2 format data" << std::endl;
        }
        return false;
    }

    cv::Mat img;
    cv::Mat bgr_mat(1536, 1920, CV_8UC3, cuda_out_buffer);
    
    /* Undistort: CUDA boost */
    cv::Mat undist_bgr_mat;
    CudaUndistort(bgr_mat, undist_bgr_mat, this->mConfig.mtx, this->mConfig.dist);
    /* Undistort */ 
    // cv::undistort(bgr_mat, undist_bgr_mat, this->mConfig.mtx, this->mConfig.dist);

    if (buffer_size == 1920*1536*2 && cuda_out_buffer != nullptr) {
        free(cuda_out_buffer);
        cuda_out_buffer = nullptr;
    }

    /* Crop to 1920x1080 */ 
    cv::Mat infer_mat = undist_bgr_mat(roi);

    /* Resize to 1280x720 */ 
    cv::resize(infer_mat, infer_mat, cv::Size(1280, 720)); 

    // engine.mInferPoints2.clear();
    // engine.mInferPoints3.clear();

    // img_out = img.clone();
    
    samplesCommon::BufferManager buffers(mEngine);
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    // clock_t start=clock();

    if (!processInput(buffers, infer_mat))
    {
        return false;
    }
    // clock_t end=clock();
    // std::cout << "preprocessing time: " << double(end-start)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    // Memcpy from host input buffers to device input buffers
    // clock_t start_ref=clock();

    buffers.copyInputToDevice();
    // mContext->setOptimizationProfileAsync(0);
    bool status = mContext->executeV2(buffers.getDeviceBindings().data());

    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // clock_t end_ref=clock();
    // std::cout << "infer time(include mem cp): " << double(end_ref-start_ref)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    
    // clock_t start_post=clock();
    // Verify results
    if (!verifyOutput(buffers, infer_mat))
    {
        return false;
    }
    // clock_t end_post=clock();

    if (this->mConfig.debug_level == DEBUG_LEVEL::DEBUG_ALL) {
        cv::imshow("camera", infer_mat);
        cv::waitKey(1);        
    }


    get_infer_points_3D(out_points);
    
    return true;
}

bool UFLD_model::rgb_normalize(const cv::Mat ori, cv::Mat &normlize, int crop_h)
{
    int normal_start = ori.rows-crop_h;

    for(int i= normal_start; i< ori.rows;i++)
    {
        for(int j=0; j< ori.cols; j++)
        {
            normlize.at<cv::Vec3f>(i-normal_start,j)[0]=(ori.at<cv::Vec3f>(i,j)[0]-0.485)/0.229;
            normlize.at<cv::Vec3f>(i-normal_start,j)[1]=(ori.at<cv::Vec3f>(i,j)[1]-0.456)/0.224;
            normlize.at<cv::Vec3f>(i-normal_start,j)[2]=(ori.at<cv::Vec3f>(i,j)[2]-0.406)/0.225;
        }
    }
    return true;
}

bool UFLD_model::each_row_max_loc(cv::Mat ori, std::vector<int> &maxloc)
{
    for(int i = 0; i< ori.rows; i++)
    {
        float *row_begin = ori.ptr<float>(i);
        int max_index = std::max_element(row_begin, row_begin+ori.cols-1)-row_begin;
        maxloc.push_back(max_index);
    }
    if(maxloc.size()!=ori.rows)
        return false;
    return true;
}

bool UFLD_model::each_row_max_loc_exist(cv::Mat ori, std::vector<int> &maxloc)
{
    for(int i = 0; i< ori.rows; i++)
    {
        float *row_begin = ori.ptr<float>(i);
        int max_index = row_begin[0]>row_begin[1]?0:1;
        maxloc.push_back(max_index);
    }
    if(maxloc.size()!=ori.rows)
        return false;
    return true;
}

void UFLD_model::print_test(float *a, int num)
{
    for(int i =0; i< num; i++)
    {
        printf("%f ",a[i]);
        if(i%16==0 && i!=0)
        {
            printf("\n");
        }
    }
    printf("\n");
}

bool UFLD_model::softmax(std::vector<float> input, std::vector<float> &re)
{
    float sum{0.0f};
    for (auto each: input)
    {
        float each_e= exp(each);
        sum += each_e;
    }
    // print_test(input.data(),3);
    for(auto each: input)
    {
        float each_e =exp(each);
        re.push_back(each_e/sum);
    }
    // print_test(re.data(),3);

    return true;


}

std::vector<double> UFLD_model::linspace(double start_in, double end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

void UFLD_model::draw_line(cv::Mat &ori,const std::vector<cv::Point> lane_ps)
{
    // printf("draw point num: %d",lane_ps.size());
    for(auto each_p: lane_ps)
    {
        // printf("x: %d, y: %d\n",each_p.x,each_p.y);
        cv::circle(ori,each_p,3,cv::Scalar(0,0,255),cv::FILLED);
    }
}

void UFLD_model::print_test(uchar *a, int num)
{
    for(int i =0; i< num; i++)
    {
        printf("%d ",(int*)a[i]);
        if(i%16==0 && i!=0)
        {
            printf("\n");
        }
    }
    printf("\n");
    

}

void UFLD_model::print_test(const std::vector<int> a,int num)
{
    for(int i =0; i< num; i++)
    {
        printf("%d ",a[i]);
        if(i%16==0 && i!=0)
        {
            printf("\n");
        }
    }
    printf("\n");
    

}

void UFLD_model::print_mat_size(const cv::Mat a)
{

    printf("w=%d, h=%d\n",a.cols,a.rows);
    
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool UFLD_model::processInput(const samplesCommon::BufferManager& buffers, cv::Mat bgr_img)
{
    const int inputH = 320;
    const int inputW = 1600;
    const int inputChannel = 3;
    
    float crop_ratio = 0.6;
    // cv::cuda::GpuMat d_bgr_img(bgr_img);

    // clock_t s1 = clock();
    std::vector<float> fileData(inputChannel * inputH * inputW);
    // clock_t e1 = clock();
    // std::cout << "preprocessing time1: " << double(e1-s1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;


    cv::cuda::GpuMat d_bgr_img(bgr_img);
    // clock_t s2 = clock();
    // cv::resize(bgr_img, bgr_img, cv::Size(inputW,int(inputH/crop_ratio)));  //resize
    cv::cuda::resize(d_bgr_img, d_bgr_img, cv::Size(inputW, inputH / crop_ratio));
    // clock_t e2 = clock();
    // std::cout << "preprocessing time2: " << double(e2-s2)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    
    // print_test(bgr_img.data,500);
    // to tensor
    // clock_t s3 = clock();
    // cv::Mat rgb_img;
    // cv::cvtColor(bgr_img,rgb_img,cv::COLOR_BGR2RGB);
    cv::cuda::GpuMat rgb_img;
    cv::cuda::cvtColor(d_bgr_img, rgb_img, cv::COLOR_BGR2RGB);
    // clock_t e3 = clock();
    // std::cout << "preprocessing time3: " << double(e3-s3)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;

    // print_test(rgb_img.data,500);
  
    // clock_t s4 = clock();
    // cv::Mat rgb_img_f;
    // rgb_img.convertTo(rgb_img_f,CV_32FC3,1.0f/255.0);
    cv::cuda::GpuMat rgb_img_f;
    rgb_img.convertTo(rgb_img_f, CV_32FC3, 1.0f / 255.0f);
    // printf("ori RGB normal 1 input\n");
    // print_mat_size(rgb_img_f);
    // print_test((float*)(rgb_img_f.data),500);
      
    //normalize 
    cv::Mat rgb_img_f_cpu;
    rgb_img_f.download(rgb_img_f_cpu);
    cv::Mat normalize(inputH, inputW, CV_32FC3); 
    rgb_normalize(rgb_img_f_cpu, normalize, inputH);

    // clock_t e4 = clock();
    // std::cout << "preprocessing time4: " << double(e4-s4)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;

    // printf("ori RGB normal 2 input\n");
    // print_mat_size(normalize);
    // print_test((float*)(normalize.data),500);
    ///UP verified 
    
    ///HWC -> CHW
    // clock_t s5 = clock();
    std::vector<cv::Mat_<float>> input_channels(inputChannel);
    cv::split(normalize, input_channels);
    // clock_t e5 = clock();

    
    // clock_t s6 = clock();
    float* data = fileData.data();  
    int channelLength = inputH * inputW;

    for(int i=0; i<3; i++)
    {
        memcpy(data,(float*)(input_channels[i].data), channelLength * sizeof(float));  //channel 
        data+=channelLength;
    }
    // clock_t e6 = clock();
    // std::cout << "preprocessing time6: " << double(e6-s6)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    // printf("input trt buff 0 0 0 : %f\n", (data[0]));
    // printf("input trt buff 0 100 100 : %f\n", data[100*inputW+100]);
    // printf("input trt buff 0 200 200 : %f\n", data[200*inputW+200]);

    // // Print an ascii representation
    // sample::gLogInfo << "Input:" << std::endl;
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    // }
    // sample::gLogInfo << std::endl;
    // clock_t s7 = clock();
    int channel_index=-1;
    int j=0;
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    //std::cout << "getHostBuffer(mParams.inputTensorNames[0]" <<buffers.getHostBuffer(mParams.inputTensorNames[0]) <<std::endl;
    for (int i = 0; i < inputChannel*inputH * inputW; i++) //normalize the image
    {
        if(i%(inputH * inputW)==0)
        {
            channel_index+=1;
        }
        j=i-channel_index*inputH * inputW;
        // hostDataBuffer[i] = fileData[i];
        hostDataBuffer[i] = ((float*)(input_channels[channel_index].data))[j];
    }
    // clock_t e7 = clock();
    // std::cout << "preprocessing time7: " << double(e7-s7)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
    // printf("input trt buff 0 0 0 : %f\n", (hostDataBuffer[0]));
    // printf("input trt buff 0 100 100 : %f\n", hostDataBuffer[100*inputW+100]);
    // printf("input trt buff 0 200 200 : %f\n", hostDataBuffer[200*inputW+200]);
    //std::cout<<"preexcution over"<<std::endl;

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool UFLD_model::verifyOutput(const samplesCommon::BufferManager& buffers, cv::Mat &img_out)
{
    float* output0 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float* output1 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
    float* output2 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
    float* output3 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[3]));
    int n_row_cls=72;
    int n_col_cls=81;
    int row_grid=200;
    int col_grid=100;
    // int original_image_height = 590;
    // int original_image_width = 1640;
    int original_image_height = 720;
    int original_image_width = 1280;
    
    cv::Mat loc_row_1= cv::Mat::zeros(n_row_cls,row_grid, CV_32FC1);  //72*200
    cv::Mat loc_row_2= cv::Mat::zeros(n_row_cls,row_grid, CV_32FC1);  //72*200

    cv::Mat loc_col_0= cv::Mat::zeros(n_col_cls,col_grid, CV_32FC1);  //81*100
    cv::Mat loc_col_3= cv::Mat::zeros(n_col_cls,col_grid, CV_32FC1);  //81*100

    
    //row calc
    for(int i =1; i< row_grid*n_row_cls*4;i+=4)
    {
        int col=i/(72*4);
        int row=((i-1)/4)%72;
        loc_row_1.at<float>(row,col)=output0[i];
        loc_row_2.at<float>(row,col)=output0[i+1];

    }

    //col calc
    for(int i=0; i< col_grid*n_col_cls*4;i+=4)
    {
        int col=i/(81*4);
        int row=(i/4)%81;
        loc_col_0.at<float>(row,col)=output1[i];
        loc_col_3.at<float>(row,col)=output1[i+3];
    }

    //std::cout << "loc_row_1" <<loc_row_1.at<float>(0,0)<<", "<<loc_row_1.at<float>(0,1)<<std::endl;
    //std::cout <<"anchor a"<<std::endl;

    //get each col max loc
    std::vector<int> max_indice_row_1;
    std::vector<int> max_indice_row_2;

    std::vector<int> max_indice_col_0;
    std::vector<int> max_indice_col_3;

    each_row_max_loc(loc_row_1,max_indice_row_1);  //72
    each_row_max_loc(loc_row_2,max_indice_row_2);  //72

    each_row_max_loc(loc_col_0,max_indice_col_0);  //81
    each_row_max_loc(loc_col_3,max_indice_col_3);  //81

    cv::Mat exist_row_1= cv::Mat::zeros(n_row_cls,2, CV_32FC1);  //72*2
    cv::Mat exist_row_2= cv::Mat::zeros(n_row_cls,2, CV_32FC1);  //72*2

    cv::Mat exist_col_0= cv::Mat::zeros(n_col_cls,2, CV_32FC1);  //81*2
    cv::Mat exist_col_3= cv::Mat::zeros(n_col_cls,2, CV_32FC1);  //81*2   

    for(int i=1; i<2*n_row_cls*4;i+=4)
    {
        int col=i/(72*4);
        int row=((i-1)/4)%72;
        exist_row_1.at<float>(row,col)=output2[i];
        exist_row_2.at<float>(row,col)=output2[i+1];
    }

    for(int i =0;i<2*n_col_cls*4;i+=4)
    {
        int col=i/(81*4);
        int row =(i/4)%81;
        exist_col_0.at<float>(row,col)=output3[i];
        exist_col_3.at<float>(row,col)=output3[i+3];
    }

    std::vector<int> valid_row_1;
    std::vector<int> valid_row_2;

    std::vector<int> valid_col_0;
    std::vector<int> valid_col_3;

    each_row_max_loc_exist(exist_row_1,valid_row_1);  //72
    each_row_max_loc_exist(exist_row_2,valid_row_2);  //72

    each_row_max_loc_exist(exist_col_0,valid_col_0);   //81
    each_row_max_loc_exist(exist_col_3,valid_col_3);   //81

    int sum_valid_row_1 = std::accumulate(valid_row_1.begin(),valid_row_1.end(),0);
    int sum_valid_row_2 = std::accumulate(valid_row_2.begin(),valid_row_2.end(),0);

    int sum_valid_col_0 = std::accumulate(valid_col_0.begin(),valid_col_0.end(),0);
    int sum_valid_col_3 = std::accumulate(valid_col_3.begin(),valid_col_3.end(),0);

    std::vector<double> row_anchor = linspace(0.42,1.0, n_row_cls);
    std::vector<double> col_anchor = linspace(0.0,1.0, n_col_cls);

    //cal row 1
    std::vector<int> row_lane_index={1,2};

    std::vector<cv::Point> pos_row_1;
    std::vector<cv::Point> pos_row_2;

    std::vector<cv::Point> pos_col_0;
    std::vector<cv::Point> pos_col_3;

    if(sum_valid_row_1>n_row_cls/2)
    {
        for(int i = 0; i < n_row_cls;i++)
        {
            if(valid_row_1[i])
            {
                int loc=max_indice_row_1[i];
                int loc_f=std::max(0,loc-1);
                int loc_b=std::min(loc+1,row_grid-1);
                //std::cout <<"anchor d"<<std::endl;

                float pre_loc= loc_row_1.at<float>(i,loc);
                float pre_loc_f= loc_row_1.at<float>(i,loc_f);
                float pre_loc_b= loc_row_1.at<float>(i,loc_b);
                std::vector<float> pre_sel={pre_loc,pre_loc_f,pre_loc_b};
                std::vector<float> soft_re;

                //std::cout <<"anchor e"<<std::endl;

                softmax(pre_sel,soft_re);
                //std::cout <<"anchor f"<<std::endl;
                // print_test(soft_re.data(), 3);
                
                float potential_row_f = loc*soft_re[0]+loc_f*soft_re[1]+loc_b*soft_re[2]+0.5;
                potential_row_f = potential_row_f/(row_grid-1) * original_image_width;
                pos_row_1.push_back(cv::Point(int(potential_row_f),int(row_anchor[i]*original_image_height)));
            }
        }

    }
    
    if(sum_valid_row_2>n_row_cls/2)
    {
        for(int i = 0; i < n_row_cls;i++)
        {
            if(valid_row_2[i])
            {
                int loc=max_indice_row_2[i];
                int loc_f=std::max(0,loc-1);
                int loc_b=std::min(loc+1,row_grid-1);
                //std::cout <<"anchor d"<<std::endl;

                float pre_loc= loc_row_2.at<float>(i,loc);
                float pre_loc_f= loc_row_2.at<float>(i,loc_f);
                float pre_loc_b= loc_row_2.at<float>(i,loc_b);
                std::vector<float> pre_sel={pre_loc,pre_loc_f,pre_loc_b};
                std::vector<float> soft_re;

                //std::cout <<"anchor e"<<std::endl;

                softmax(pre_sel,soft_re);
                //std::cout <<"anchor f"<<std::endl;
                // print_test(soft_re.data(), 3);
                
                float potential_row_f = loc*soft_re[0]+loc_f*soft_re[1]+loc_b*soft_re[2]+0.5;
                //std::cout <<"anchor f1"<<std::endl;
                potential_row_f = potential_row_f/(row_grid-1) * original_image_width;
                //std::cout <<"anchor f2"<<std::endl;
                pos_row_2.push_back(cv::Point(int(potential_row_f),int(row_anchor[i]*original_image_height)));
                //std::cout <<"total valid line: "<<pos_row_2.size()<<std::endl;
            }
        }

    }
    //col 0
    if(sum_valid_col_0>n_col_cls/4)
    {
        for(int i = 0; i < n_col_cls;i++)
        {
            if(valid_col_0[i])
            {
                int loc=max_indice_col_0[i];
                int loc_f=std::max(0,loc-1);
                int loc_b=std::min(loc+1,col_grid-1);
                //std::cout <<"anchor d"<<std::endl;

                float pre_loc= loc_col_0.at<float>(i,loc);
                float pre_loc_f= loc_col_0.at<float>(i,loc_f);
                float pre_loc_b= loc_col_0.at<float>(i,loc_b);
                std::vector<float> pre_sel={pre_loc,pre_loc_f,pre_loc_b};
                std::vector<float> soft_re;

                //std::cout <<"anchor e"<<std::endl;

                softmax(pre_sel,soft_re);
                //std::cout <<"anchor f"<<std::endl;
                // print_test(soft_re.data(), 3);
                
                float potential_col_f = loc*soft_re[0]+loc_f*soft_re[1]+loc_b*soft_re[2]+0.5;
                //std::cout <<"anchor f1"<<std::endl;
                potential_col_f = potential_col_f/(col_grid-1) * original_image_height;
                //std::cout <<"anchor f2"<<std::endl;
                pos_col_0.push_back(cv::Point(int(col_anchor[i]*original_image_width),int(potential_col_f)));
                //std::cout <<"total valid col 0 line: "<<pos_col_0.size()<<std::endl;
                std::reverse(pos_col_0.begin(), pos_col_0.end());
            }
        }
    }

    //col 3
    if(sum_valid_col_3>n_col_cls/4)
    {
        for(int i = 0; i < n_col_cls;i++)
        {
            if(valid_col_3[i])
            {
                int loc=max_indice_col_3[i];
                int loc_f=std::max(0,loc-1);
                int loc_b=std::min(loc+1,col_grid-1);
                //std::cout <<"anchor d"<<std::endl;

                float pre_loc= loc_col_3.at<float>(i,loc);
                float pre_loc_f= loc_col_3.at<float>(i,loc_f);
                float pre_loc_b= loc_col_3.at<float>(i,loc_b);
                std::vector<float> pre_sel={pre_loc,pre_loc_f,pre_loc_b};
                std::vector<float> soft_re;

                //std::cout <<"anchor e"<<std::endl;

                softmax(pre_sel,soft_re);
                //std::cout <<"anchor f"<<std::endl;
                // print_test(soft_re.data(), 3);
                
                float potential_col_f = loc*soft_re[0]+loc_f*soft_re[1]+loc_b*soft_re[2]+0.5;
                //std::cout <<"anchor f1"<<std::endl;
                potential_col_f = potential_col_f/(col_grid-1) * original_image_height;
                //std::cout <<"anchor f2"<<std::endl;
                pos_col_3.push_back(cv::Point(int(col_anchor[i]*original_image_width),int(potential_col_f)));
                //std::cout <<"total valid col 0 line: "<<pos_col_3.size()<<std::endl;
            }
        }

    }

    //clc col
    cv::Mat ori_img =  img_out;
    draw_line(ori_img, pos_col_0);
    draw_line(ori_img, pos_row_1);
    draw_line(ori_img, pos_row_2);
    draw_line(ori_img, pos_col_3);

    mInferPoints2.push_back(pos_col_0);
    mInferPoints2.push_back(pos_row_1);
    mInferPoints2.push_back(pos_row_2);
    mInferPoints2.push_back(pos_col_3);

    return true;
}

void UFLD_model::heightToPointCloud(int u, int v, const Config &config, cv::Point3f& point) 
{
    // Camera intrinsics, standrad: 1920*1080
    // Eigen::Matrix3f camera_intrinsics;
    // camera_intrinsics << 1009.697789, 0, 992.376835,
    //                      0, 1009.573423, 759.462196 - 228,
    //                      0, 0, 1;

    float fx = static_cast<float>(config.mtx.at<double>(0,0));
    float fy = static_cast<float>(config.mtx.at<double>(1,1));
    float cx = static_cast<float>(config.mtx.at<double>(0,2));
    float cy = static_cast<float>(config.mtx.at<double>(1,2)) - roi.y;
    // std::cout << "fx: " << fx << std::endl; 
    // std::cout << "fy: " << fy << std::endl; 
    // std::cout << "cx: " << cx << std::endl; 
    // std::cout << "cy: " << cy << std::endl; 
    // std::cout << "roi.y: " << roi.y << std::endl;
    // float v0 = 1080 / 2;

    float y3d = config.camera_height/static_cast<float>(100);
    // std::cout << __LINE__ << ":" << y3d << std::endl;
    // std::cout << __LINE__ << ":" << config.camera_height << std::endl;
    float z3d;
    if (v > cy) {
        z3d = y3d * fy / float(v - cy);
    } else {
        throw std::invalid_argument("Invalid value v"); 
    }

    float x3d = (u - cx) * z3d / fx;

    uint depth_max = 100;
    if (z3d == 0 || z3d >= depth_max) {
        throw std::invalid_argument("Invalid depth value");

    }

    point.x = x3d;
    point.y = y3d;
    point.z = z3d;
    // std::cout << __LINE__ << ":" << point.x << ", " << point.y << ", " << point.z << std::endl; 
}

void UFLD_model::point2to3(const Config &config, const std::vector<cv::Point2i>& points2, std::vector<cv::Point3i>& points3)
{
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    if (config.camera_height == -1 && this->mConfig.debug_level >= DEBUG_LEVEL::DEBUG_ERROR) {
        std::cerr << __FILE__ << ":" << __LINE__ << " error camera height, please input camera_height in Config" << std::endl;
        return ;
    }
    for (const auto& point : points2) {
        cv::Point3f point_3D;
        try {
            // 2d to 3d, image 2d point(u,v), fixed height=1.46m
            // 1280*1.5=1920, 720*1.5=1080
            // max depth 100m
            heightToPointCloud(point.x*1.5, point.y*1.5, config, point_3D);
            // std::cout << "point.x: " << point.x << ", " << "point.y: " << point.y << std::endl;
        } catch (const std::exception& e) {
            if (this->mConfig.debug_level >= DEBUG_LEVEL::DEBUG_ERROR) {
                std::cerr << __FILE__ << ":" << __LINE__ << " " << e.what() << std::endl;
            }
        }

        // unit: m -> cm, coordinate system: -> UE
        int UE_camera_x = point_3D.z*100;
        int UE_camera_y = point_3D.x*100;
        int UE_camera_z = point_3D.y*100; 

        // coordinate system: camera -> world
        int UE_world_x = UE_camera_x - config.ue_camera_position.at<int>(0,0);
        int UE_world_y = UE_camera_y - config.ue_camera_position.at<int>(0,1);
        int UE_world_z = UE_camera_z - config.ue_camera_position.at<int>(0,2);

        cv::Point3i point3;
        point3.x = UE_world_x;  
        point3.y = UE_world_y; 
        point3.z = UE_world_z;  

        points3.push_back(point3);
        // std::cout << "point3.x: " << point3.x << " " << "point3.y: " << point3.y <<  " " << "point3.z: " << point3.z << std::endl;
        
    }
}


void UFLD_model::get_infer_points_2D(std::vector<std::vector<cv::Point2i>> &points)
{
    points = mInferPoints2;
    // for (auto points_ : points) {
    //     for (auto point : points_) {
            
    //         std::cout << point.x << "," << point.y << "," << point.z << "  ";
    //     }
    //     std::cout << std::endl;
    // }
}

void UFLD_model::get_infer_points_3D(std::vector<std::vector<cv::Point3i>> &points)
{
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    /* confirm mInferPoints2 */
    // std::cout << "get_infer_points_3D:" << std::endl;
    // for (auto points_ : mInferPoints2) {
    //     std::cout << "mInferPoints2" << std::endl;
    //     for (auto point : points_) {
    //         std::cout << point.x << "," << point.y << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<cv::Point3i> point3_0(0);
    std::vector<cv::Point3i> point3_1(0);
    std::vector<cv::Point3i> point3_2(0);
    std::vector<cv::Point3i> point3_3(0);
    
    if (mInferPoints2[0].size() > 0) {
        point2to3(mConfig, mInferPoints2[0], point3_0); 
    }
    if (mInferPoints2[1].size() > 0) {
        point2to3(mConfig, mInferPoints2[1], point3_1); 
    }
    if (mInferPoints2[2].size() > 0) {
        point2to3(mConfig, mInferPoints2[2], point3_2);
    }
    if (mInferPoints2[3].size() > 0) {
        point2to3(mConfig, mInferPoints2[3], point3_3); 
    }
    
    mInferPoints3.push_back(point3_0);
    mInferPoints3.push_back(point3_1);
    mInferPoints3.push_back(point3_2);
    mInferPoints3.push_back(point3_3);

    points = mInferPoints3;

    /* confirm 3d points */
    // int vecIndex = 0;
    // for (const auto& vec : points)
    // {
    //     std::cout << "Vector " << vecIndex++ << ":" << std::endl;
    //     int pointIndex = 0;
    //     for (const auto& point : vec)
    //     {
    //         std::cout << point;
    //     }
    //     std::cout << std::endl;
    // }
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams()
{
    samplesCommon::Args args;
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.push_back("data/mnist/");
    params.dataDirs.push_back("data/samples/mnist/");
    params.onnxFileName = "culane_res18.onnx";
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("loc_row");
    params.outputTensorNames.push_back("loc_col");
    params.outputTensorNames.push_back("exist_row");
    params.outputTensorNames.push_back("exist_col");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}
