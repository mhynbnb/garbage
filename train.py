from ultralytics import YOLO
if __name__ == '__main__':

# Load a model
    model = YOLO('yolov8n.pt')
#     model=YOLO('yolov81.yaml')

#     model = YOLO(r'runs\obb\train4\weights\last.pt')

    # Train the model
    results = model.train( data='./garbage.yaml',cos_lr=True,epochs=100, imgsz=640,batch=32,lr0=0.01)
    # results = model.train( data='Brackish.yaml',cos_lr=True,epochs=400, imgsz=800,batch=-1,lr0=0.01)