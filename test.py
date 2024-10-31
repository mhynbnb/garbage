'''
File : test.py
Auther : MHY
Created : 2024/6/24 20:07
Last Updated : 
Description : 
Version : 
'''
from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train/weights/best.pt')

# Customize validation settings
validation_results = model.val(data="./garbage.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)