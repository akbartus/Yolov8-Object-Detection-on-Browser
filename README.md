<p align="center">
  <img src="img/screenshot.jpg" />
</p>


This is adapted and reduced version of YOLOv8 object segmentation (powered by onnx) created by <a href="https://github.com/Hyuto/yolov8-onnxruntime-web">Wahyu Setianto</a>. This version can be run on JavaScript without any frameworks and demonstrates object detection using web camera.

## Setup
To see it at work, just run index.html file. 

## Models

**Main Model**

YOLOv8n model converted to onnx with input dimensions of 416x416. 

```
used model : yolov8n.onnx
size       : ~ 12.5Mb
```

**NMS**

ONNX model to perform NMS operator [CUSTOM].

```
nms-yolov8.onnx
```


## Use another model

It is possible to use bigger models converted to onnx, however this might impact the total loading time.

To use another YOLOv8 model, download it from Ultralytics and convert it to onnx file format.

**Custom YOLOv8 Object Detection Models**

Please update labels object inside of main.js file.


## Demo
To see demo, please visit the <a href="https://yolov8-segmentation.glitch.me/">following page</a>
