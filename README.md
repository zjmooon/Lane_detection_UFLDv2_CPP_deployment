## Description

### Lane detection module
1. Perform lane detection on the input image based on [Ultra Fast Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection).
2. Selecting different APIs can get different types of results, including 2D and 3D points.
#### NOTE:
1. The maximum number of lanes detected is 4.
2. Each lane is stored in a separate array.

### API
> infer(void* buffer, size_t buffer_size, ...)

The buffer can come from the camera or the video:
- The data type of the raw data needs to be unsigned char*. 
- The size of the raw data should be 1920*1536 and not calibrated.

#### NOTE:
 Currently only support two data format, the API has been adapted internally
-  camera: buffer_size=1920\*1536\*2 (YUYV)
-  video :  buffer_size=1920\*1536\*3 (BGR)


## Build
``` shell 
mkdir build
cd build
cmake ..
make
```

## Run

Usage:<br>
```
./lane_detection
```


**Test Enviroment**
```
Jetson AGX Orin 32GB
Jetpack 5.1.2
CUDA 11.4
OpenCV with CUDA 4.4.0
```


