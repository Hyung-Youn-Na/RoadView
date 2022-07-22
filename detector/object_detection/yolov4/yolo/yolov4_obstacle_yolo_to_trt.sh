python3 yolo_to_onnx.py -m yolov4-416 -c 15
python3 onnx_to_tensorrt.py -m yolov4-416 -c 15 --verbose