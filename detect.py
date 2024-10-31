'''
File : detect.py
Auther : MHY
Created : 2024/3/20 15:54
Last Updated : 
Description : 
Version : 
'''
import os
import time

from ultralytics import YOLO
model = YOLO('yolov8x-worldv2.pt') 
# model = YOLO('./runs/detect/train7/weights/best.pt')
# model.set_classes(["bone"])

model.predict(r"E:\Download\Garbage\kitchen\bone", save=True, imgsz=800, conf=0.25,save_txt=True,show=False)
# model.predict(r'0', save=True, imgsz=800, conf=0.5,save_txt=True,show=True)

