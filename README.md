# OpenCV Zoo and Benchmark

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:

- Install latest `opencv-python`:
  ```shell
  python3 -m pip install opencv-python
  # Or upgrade to latest version
  python3 -m pip install --upgrade opencv-python
  ```
- Clone this repo to download all models and demo scripts:
  ```shell
  # Install git-lfs from https://git-lfs.github.com/
  git clone https://github.com/opencv/opencv_zoo && cd opencv_zoo
  git lfs install
  git lfs pull
  ```
- To run benchmarks on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).

## Models & Benchmark Results

| Model                                                   | Task                          | Input Size | CPU-INTEL (ms) | CPU-RPI (ms) | GPU-JETSON (ms) | NPU-KV3 (ms) | NPU-Ascend310 (ms) | CPU-D1 (ms) |
| ------------------------------------------------------- | ----------------------------- | ---------- | -------------- | ------------ | --------------- | ------------ | ------------------ | ----------- |
| [YuNet](./models/face_detection_yunet)                  | Face Detection                | 160x120    | 0.72           | 5.43         | 12.18           | 4.04         | 2.24               | 86.69       |
| [SFace](./models/face_recognition_sface)                | Face Recognition              | 112x112    | 6.04           | 78.83        | 24.88           | 46.25        | 2.66               | ---         |
| [FER](./models/facial_expression_recognition/)          | Facial Expression Recognition | 112x112    | 3.16           | 32.53        | 31.07           | 29.80        | 2.19               | ---         |
| [LPD-YuNet](./models/license_plate_detection_yunet/)    | License Plate Detection       | 320x240    | 8.63           | 167.70       | 56.12           | 29.53        | 7.63               | ---         |
| [YOLOX](./models/object_detection_yolox/)               | Object Detection              | 640x640    | 141.20         | 1805.87      | 388.95          | 420.98       | 28.59              | ---         |
| [NanoDet](./models/object_detection_nanodet/)           | Object Detection              | 416x416    | 66.03          | 225.10       | 64.94           | 116.64       | 20.62              | ---         |
| [DB-IC15](./models/text_detection_db) (EN)              | Text Detection                | 640x480    | 71.03          | 1862.75      | 208.41          | ---          | 17.15              | ---         |
| [DB-TD500](./models/text_detection_db) (EN&CN)          | Text Detection                | 640x480    | 72.31          | 1878.45      | 210.51          | ---          | 17.95              | ---         |
| [CRNN-EN](./models/text_recognition_crnn)               | Text Recognition              | 100x32     | 20.16          | 278.11       | 196.15          | 125.30       | ---                | ---         |
| [CRNN-CN](./models/text_recognition_crnn)               | Text Recognition              | 100x32     | 23.07          | 297.48       | 239.76          | 166.79       | ---                | ---         |
| [PP-ResNet](./models/image_classification_ppresnet)     | Image Classification          | 224x224    | 34.71          | 463.93       | 98.64           | 75.45        | 6.99               | ---         |
| [MobileNet-V1](./models/image_classification_mobilenet) | Image Classification          | 224x224    | 5.90           | 72.33        | 33.18           | 145.66\*     | 5.15               | ---         |
| [MobileNet-V2](./models/image_classification_mobilenet) | Image Classification          | 224x224    | 5.97           | 66.56        | 31.92           | 146.31\*     | 5.41               | ---         |
| [PP-HumanSeg](./models/human_segmentation_pphumanseg)   | Human Segmentation            | 192x192    | 8.81           | 73.13        | 67.97           | 74.77        | 6.94               | ---         |
| [WeChatQRCode](./models/qrcode_wechatqrcode)            | QR Code Detection and Parsing | 100x100    | 1.29           | 5.71         | ---             | ---          | ---                | ---         |
| [DaSiamRPN](./models/object_tracking_dasiamrpn)         | Object Tracking               | 1280x720   | 29.05          | 712.94       | 76.82           | ---          | ---                | ---         |
| [YoutuReID](./models/person_reid_youtureid)             | Person Re-Identification      | 128x256    | 30.39          | 625.56       | 90.07           | 44.61        | 5.58               | ---         |
| [MP-PalmDet](./models/palm_detection_mediapipe)         | Palm Detection                | 192x192    | 6.29           | 86.83        | 83.20           | 33.81        | 5.17               | ---         |
| [MP-HandPose](./models/handpose_estimation_mediapipe)   | Hand Pose Estimation          | 224x224    | 4.68           | 43.57        | 40.10           | 19.47        | 6.27               | ---         |

\*: Models are quantized in per-channel mode, which run slower than per-tensor quantized models on NPU.

Hardware Setup:

- `CPU-INTEL`: [Intel Core i7-12700K](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html), 8 Performance-cores (3.60 GHz, turbo up to 4.90 GHz), 4 Efficient-cores (2.70 GHz, turbo up to 3.80 GHz), 20 threads.
- `CPU-RPI`: [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/), Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5 GHz.
- `GPU-JETSON`: [NVIDIA Jetson Nano B01](https://developer.nvidia.com/embedded/jetson-nano-developer-kit), 128-core NVIDIA Maxwell GPU.
- `NPU-KV3`: [Khadas VIM3](https://www.khadas.com/vim3), 5TOPS Performance. Benchmarks are done using **quantized** models. You will need to compile OpenCV with TIM-VX following [this guide](https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU) to run benchmarks. The test results use the `per-tensor` quantization model by default.
- `NPU-Ascend310`: [Ascend 310](https://e.huawei.com/uk/products/cloud-computing-dc/atlas/atlas-200), 22 TOPS @ INT8. Benchmarks are done on [Atlas 200 DK AI Developer Kit](https://e.huawei.com/in/products/cloud-computing-dc/atlas/atlas-200). Get the latest OpenCV source code and build following [this guide](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) to enable CANN backend.
- `CPU-D1`: [Allwinner D1](https://d1.docs.aw-ol.com/en), [Xuantie C906 CPU](https://www.t-head.cn/product/C906?spm=a2ouz.12986968.0.0.7bfc1384auGNPZ) (RISC-V, RVV 0.7.1) @ 1.0 GHz, 1 core. YuNet is supported for now. Visit [here](https://github.com/fengyuentau/opencv_zoo_cpp) for more details.

***Important Notes***:

- The data under each column of hardware setups on the above table represents the elapsed time of an inference (preprocess, forward and postprocess).
- The time data is the mean of 10 runs after some warmup runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- `---` represents the model is not availble to run on the device.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

## Some Examples

Some examples are listed below. You can find more in the directory of each model!

### Face Detection with [YuNet](./models/face_detection_yunet/)

![largest selfie](./models/face_detection_yunet/examples/largest_selfie.jpg)

### Facial Expression Recognition with [Progressive Teacher](./models/facial_expression_recognition/)

![fer demo](./models/facial_expression_recognition/examples/selfie.jpg)

### Human Segmentation with [PP-HumanSeg](./models/human_segmentation_pphumanseg/)

![messi](./models/human_segmentation_pphumanseg/examples/messi.jpg)

### License Plate Detection with [LPD_YuNet](./models/license_plate_detection_yunet/)

![license plate detection](./models/license_plate_detection_yunet/examples/lpd_yunet_demo.gif)

### Object Detection with [NanoDet](./models/object_detection_nanodet/) & [YOLOX](./models/object_detection_yolox/)

![nanodet demo](./models/object_detection_nanodet/samples/1_res.jpg)

![yolox demo](./models/object_detection_yolox/samples/3_res.jpg)

### Object Tracking with [DaSiamRPN](./models/object_tracking_dasiamrpn/)

![webcam demo](./models/object_tracking_dasiamrpn/examples/dasiamrpn_demo.gif)

### Palm Detection with [MP-PalmDet](./models/palm_detection_mediapipe/)

![palm det](./models/palm_detection_mediapipe/examples/mppalmdet_demo.gif)

### Hand Pose Estimation with [MP-HandPose](models/handpose_estimation_mediapipe/)

![handpose estimation](models/handpose_estimation_mediapipe/examples/mphandpose_demo.webp)

### QR Code Detection and Parsing with [WeChatQRCode](./models/qrcode_wechatqrcode/)

![qrcode](./models/qrcode_wechatqrcode/examples/wechat_qrcode_demo.gif)

### Chinese Text detection [DB](./models/text_detection_db/)

![mask](./models/text_detection_db/examples/mask.jpg)

### English Text detection [DB](./models/text_detection_db/)

![gsoc](./models/text_detection_db/examples/gsoc.jpg)

### Text Detection with [CRNN](./models/text_recognition_crnn/)

![crnn_demo](./models/text_recognition_crnn/examples/CRNNCTC.gif)














































# Running Yunet Face Detection on Jetson

## This guide will walk you through the steps to run the Yunet face detection demo on Jetson using Python 3.7.

### Prerequisites
- Git Large File Storage (LFS)
- Onnx runtime

#### Step 1: Installing Git LFS
To install Git LFS, follow these steps:

- Download Git LFS from https://git-lfs.com/.
- Run the following commands in the terminal:

```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs=3.3.0
```

#### Step 2: Cloning the opencv_zoo Repository
To clone the opencv_zoo repository, follow these steps:

- Open terminal and run the following command:
```sh
git clone https://github.com/opencv/opencv_zoo

```
- Navigate to the cloned repository using the following command:
```sh 
cd opencv_zoo
```

- Install Git LFS for the repository using the following command:
```sh 
git lfs install	

```
- Pull the LFS data for the repository using the following command:
``` sh 
git lfs pull
```


#### Step 3: Installing Onnx runtime
To install Onnx runtime, follow these steps:

- Go to the link https://elinux.org/Jetson_Zoo#ONNX_Runtime
- Download the .whl file that is compatible with your Python version.
- Install the .whl file using pip:
``` sh 
pip install <filename>.whl

```


#### Step 4: Running the Demo
To run the demo, follow these steps:

- Navigate to the face detection demo directory using the following command:
``` sh 
cd opencv_zoo/models/face_detection_yunet

```
- Run the demo using the following command:
``` sh 
python3.7 demo.py	

```

The demo will start running and display the results of the face detection in the output window.

### Note: If you encounter any issues, make sure to check that all the necessary dependencies are installed and that you have the correct version of Python installed on your system.

# YuNet

YuNet is a light-weight, fast and accurate face detection model, which achieves 0.834(AP_easy), 0.824(AP_medium), 0.708(AP_hard) on the WIDER Face validation set.

Notes:

- Model source: [here](https://github.com/ShiqiYu/libfacedetection.train/blob/a61a428929148171b488f024b5d6774f93cdbc13/tasks/task1/onnx/yunet.onnx).
- This model can detect **faces of pixels between around 10x10 to 300x300** due to the training scheme.
- For details on training this model, please visit https://github.com/ShiqiYu/libfacedetection.train.
- This ONNX model has fixed input shape, but OpenCV DNN infers on the exact shape of input image. See https://github.com/opencv/opencv_zoo/issues/44 for more information.

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models      | Easy AP | Medium AP | Hard AP |
| ----------- | ------- | --------- | ------- |
| YuNet       | 0.8498  | 0.8384    | 0.7357  |
| YuNet quant | 0.7751  | 0.8145    | 0.7312  |

\*: 'quant' stands for 'quantized'.



# Progressive Teacher (Facial Expression Recognition)

Progressive Teacher: [Boosting Facial Expression Recognition by A Semi-Supervised Progressive Teacher](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=OCwcfAwAAAAJ&citation_for_view=OCwcfAwAAAAJ:u5HHmVD_uO8C)

Note:
- Progressive Teacher is contributed by [Jing Jiang](https://scholar.google.com/citations?user=OCwcfAwAAAAJ&hl=zh-CN).
-  [MobileFaceNet](https://link.springer.com/chapter/10.1007/978-3-319-97909-0_46) is used as the backbone and the model is able to classify seven basic facial expressions (angry, disgust, fearful, happy, neutral, sad, surprised).
- [facial_expression_recognition_mobilefacenet_2022july.onnx](https://github.com/opencv/opencv_zoo/raw/master/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx) is implemented thanks to [Chengrui Wang](https://github.com/crywang).

Results of accuracy evaluation on [RAF-DB](http://whdeng.cn/RAF/model1.html).

| Models      | Accuracy | 
|-------------|----------|
| Progressive Teacher       | 88.27%  |


## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the demo:
```shell
# recognize the facial expression on images
python demo.py --input /path/to/image
```

### Example outputs

Note: Zoom in to to see the recognized facial expression in the top-left corner of each face boxes.

![fer demo](./examples/selfie.jpg)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/abstract/document/9629313






# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:

- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy).
- Model files encode MobileFaceNet instances trained on the SFace loss function, see the [SFace paper](https://arxiv.org/abs/2205.12010) for reference.
- ONNX file conversions from [original code base](https://github.com/zhongyy/SFace) thanks to [Chengrui Wang](https://github.com/crywang).
- (As of Sep 2021) Supporting 5-landmark warping for now, see below for details.

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models      | Accuracy |
| ----------- | -------- |
| SFace       | 0.9940   |
| SFace quant | 0.9932   |

\*: 'quant' stands for 'quantized'.

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace








## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
