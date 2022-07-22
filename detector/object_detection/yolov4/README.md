# yolov4-416 TensorRT 모델
* 본 문서는 Edge-module에서 사용되는 객체 검출(object detection) 모델에 대한 설명을 하기 위한 문서이다.

## 학습 데이터셋
* 본 모델은 위험 장애물 데이터셋(Obstacle-15)으로 학습을 진행하였으며, 아래 class로 구성되어 있다.
  * 데이터셋 다운로드는 관리자에게 문의
### 이미지 상세 정보
  * __Input size__ : 416x416
### 데이터셋 구성
  * 총 image, label 수 : 279,243
    * __train__: 186,163
    * __test__: 46,540
    * __val__: 46,540

| Class           | # of object |
|:---------------:|------------:|
| person          |     384,921 |
| bicycle         |      43,596 |
| bus             |      38,910 |
| car             |     673,116 |
| carrier         |       8,223 |
| motorcycle      |      50,566 |
| movable_signage |     112,651 |
| truck           |     114,994 |
| bollard         |     239,686 |
| chair           |      26,665 |
| potted_plant    |      62,121 |
| table           |       7,983 |
| tree_trunk      |     34,4375 |
| pole            |     322,610 |
| fire_hydrant    |       9,254 |

## 모델 학습
* 모델 학습은 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 로 진행
  * 학습 방법은 [https://github.com/AlexeyAB/darknet]() 참고
* 학습 후 저장되는 ```${model_name}.weights``` 파일을 weights → onnx → tensorrt 순으로 변환해야 Jetson Nano, Jetson NX Xavier에서 사용가능하다.
### 모델 성능
#### yolov4-416_weights(```${model_name}.weight```)
* __mAP__ : 0.7423
* __Speed__ : 80fps(RTX3090)

| Class           | AP     |
|:---------------:|-------:|
| person          | 0.7950 |
| bicycle         | 0.7909 |
| bus             | 0.8700 |
| car             | 0.9387 |
| carrier         | 0.4458 |
| motorcycle      | 0.8535 |
| movable_signage | 0.7625 |
| truck           | 0.8720 |
| bollard         | 0.6939 |
| chair           | 0.6450 |
| potted_plant    | 0.6551 |
| table           | 0.5652 |
| tree_trunk      | 0.7814 |
| pole            | 0.8155 |
| fire_hydrant    | 0.6798 |
#### yolov4-416_TensorRT (```${model_name}.trt```)
* __mAP__ : 0.6464
* __Speed__ : 21fps(Jetson NX Xavier)

| Class           | AP     |
|:---------------:|-------:|
| person          | 0.7305 |
| bicycle         | 0.6438 |
| bus             | 0.7819 |
| car             | 0.8805 |
| carrier         | 0.3809 |
| motorcycle      | 0.7642 |
| movable_signage | 0.6578 |
| truck           | 0.7873 |
| bollard         | 0.6984 |
| chair           | 0.5100 |
| potted_plant    | 0.5315 |
| table           | 0.4430 |
| tree_trunk      | 0.6205 |
| pole            | 0.6729 |
| fire_hydrant    | 0.5931 |

## Object Detection(yolov4-416 tensorrt) 모델 사용 방법
* 모델 디렉토리 구조는 아래와 같다.
```shell
/workspace/detector/object_detection/yolov4
├── __init__.py
├── plugins
│   ├── gpu_cc.py
│   ├── __init__.py
│   ├── libyolo_layer.so
│   ├── Makefile
│   ├── README.md
│   ├── yolo_layer.cu
│   ├── yolo_layer.h
│   └── yolo_layer.o
├── __pycache__
│   ├── __init__.cpython-36.pyc
│   └── yolov4.cpython-36.pyc
├── README.md
├── yolo
│   ├── calibrator.py
│   ├── download_yolov4_416_obstacle.sh
│   ├── __init__.py
│   ├── onnx_to_tensorrt.py
│   ├── plugins.py
│   ├── yolo_to_onnx.py
│   └── yolov4_obstacle_yolo_to_trt.sh
└── yolov4.py
```
### 모듈 사용을 위해 수행해야 할 사항
* 해당 모듈을 사용하기 위해서는 [building_on_jetson.md](https://github.com/JinhaSong/edge-analysis-module/blob/master/docs/building_on_jetson.md) 또는 [building_on_docker.md](https://github.com/JinhaSong/edge-analysis-module/blob/master/docs/building_on_docker-compose.md) 의 install-python-requirements 항목을 포함한 하위 빌드 과정을 진행해야 모델을 통해 이미지를 추론할 수 있다.
#### 수행해야 할 사항([링크](https://github.com/JinhaSong/edge-analysis-module/blob/master/docs/building_on_jetson.md) 참고)
* python requirements 설치(${WORKSPACE}/requirements.txt)
* Protobuf, PyCUDA 설치(```${WORKSPACE}/scripts``` 하위에서 진행) 
* ONNX 설치
* yolo layer 빌드(```${WORKSPACE}/detector/object_detection/yolov4/plugins``` 하위에서 진행)
* yolov4-obstacle-15 모델 다운로드 및 tensorrt 모델로 변환
### 모델 테스트
```python
# in ${WORKSPACE}
from detector.object_detection.yolov4.yolov4 import YOLOv4
import cv2

# Load model
model = YOLOv4()

# load image
image_path = "test.jpg"
image = cv2.imread(image_path)

# Inference image
result = model.inference_by_image(image)

# Print object detection result
print(result)
```