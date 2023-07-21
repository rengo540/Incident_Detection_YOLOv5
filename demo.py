import os
import cv2
import torch
import numpy as np
from tracking.tracker import *
import argparse
import pandas as pd
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from models.common import DetectMultiBackend
from models.experimental import attempt_load
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode





def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        










#model_accident = torch.hub.load('.', 'custom', path='runs/best.pt', source='local') 
device = select_device('')
imgsz=(416, 416)  # inference size (height, width)
model = DetectMultiBackend('yolov5m.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Carnames, pt = model.stride, model.names, model.pt

model_accident = DetectMultiBackend('afterSeminar/modelVer6/weights/last.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Accidentnames, pt = model_accident.stride, model_accident.names, model_accident.pt

model_flood = DetectMultiBackend('runs/train-seg/modelFlods2/best.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Floodnames, pt = model_flood.stride, model_flood.names, model_flood.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

writer= cv2.VideoWriter('output/crashOutput/dense3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1320,600))


source = str('exampleClips/dense3.mp4')
conf_thres=0.5  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
bs = 1  # batch_size
area = [(537,597),(1001,587),(722,318),(468,340)]
dense_threshold=11
#tracker = Tracker()

dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=2)



accident_count=0
fire_count=0 
flood_count=0
acc_threshold=2
fire_threshold=2
flood_threshold=4

 # Run inference
model_flood.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model_accident.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
for path, im, im0s, vid_cap, s in dataset:
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', POINTS)

    with dt[0]:
        im = torch.from_numpy(im).to(model_flood.device)
        im = im.half() if model_flood.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
 
    im0s=cv2.resize(im0s,(1020,600))
    
    #cv2.polylines(im0s,[np.array(area,np.int32)],True,(0,255,0),2)


    # Inference
    with dt[1]:
        
        floods_pred, proto = model_flood(im,  augment=False,visualize=False)[:2]
        car_pred = model(im, augment=False, visualize=False)
        accident_pred = model_accident(im, augment=False, visualize=False)

   

   
    # NMS
    with dt[2]:
        car_pred = non_max_suppression(car_pred, 0.3, iou_thres, (2,5,7), False, max_det=10000)
        accident_pred = non_max_suppression(accident_pred, 0.65, iou_thres, (0,1,2,4), False, max_det=100)
        floods_pred = non_max_suppression(floods_pred, 0.6, iou_thres, 1, False, max_det=10, nm=32)
        
    dense_count=0
    # Car predictions
    for i, det in enumerate(car_pred):  # per image
        seen += 1
        
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        
        #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                confidence_score = conf
                #class_index = cls
                object_name = Carnames[int(cls)]

                print('bounding box is ', x1, y1, x2, y2)
                print('detected object name is ', object_name)
                cv2.rectangle(im0s,(x1,y1),(x2,y2),(255,0,0),3)
                cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)         
                cv2.circle(im0s,(x2,y2),4,(0,255,0),-1)
                results = cv2.pointPolygonTest(np.array(area,np.int32), (x2,y2) ,False)
                if results>=0:
                       dense_count += 1




 # accident predictions
    
    for i, det in enumerate(accident_pred):  # per image
        seen += 1
        
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        
        #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                confidence_score = conf
                #class_index = cls
                object_name = Accidentnames[int(cls)]
                if object_name == 'fire':
                    fire_count +=1
                else:
                    accident_count+=1
                print('bounding box is ', x1, y1, x2, y2)
                print('detected object name is ', object_name)
                #points.append([x1,x2,x2,y2,con,n])
                cv2.rectangle(im0s,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)



    # floods predictions
    for i, det in enumerate(floods_pred):  # per image
        seen += 1
       
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p) 
        s += '%gx%g ' % im.shape[2:]  # print string
        
        if len(det):
            
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())

                    confidence_score = conf
                    #class_index = cls
                    object_name = Floodnames[int(cls)]
                    flood_count+=1

                    print('bounding box is ', x1, y1, x2, y2)
                    print('detected object name is ', object_name)
                    cv2.rectangle(im0s,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

    top, bottom, left, right = 0, 0, 0, 300
    border_color = [255, 255, 255]  # White color
    # Add the border
    im0s_with_border = cv2.copyMakeBorder(im0s, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

   
    cv2.putText(im0s_with_border,'INCIDENT TYPE',(1050,160),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)

    
    if  acc_threshold<accident_count:
         cv2.putText(im0s_with_border,'car accident',(1050,190),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
         acc_threshold=accident_count


    if  fire_threshold<fire_count:
        cv2.putText(im0s_with_border,'fire',(1050,190),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
        fire_threshold=fire_count

    if flood_threshold<flood_count:
        cv2.putText(im0s_with_border,'flood',(1050,190),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
        fire_threshold=fire_count
        flood_count = flood_count-flood_threshold

    cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
    cv2.putText(im0s_with_border,'Dense Threshold ='+str(dense_threshold),(1050,120),cv2.FONT_ITALIC,0.5,(0,0,0),1)                         
    if dense_threshold < dense_count:
        cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,0,255),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
        cv2.putText(im0s_with_border,'dense traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
    else:
        cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,255,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
        cv2.putText(im0s_with_border,'sparse traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
    
    cv2.imshow("img",im0s_with_border)
    writer.write(im0s_with_border)

    #print(a)
    if cv2.waitKey(1)&0xFF==27:
        break
writer.release()

cv2.destroyAllWindows()











"""
tracker = Tracker()


dense_threshold = 5


#writer= cv2.VideoWriter('tracking/deee.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1020,600))

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

area = [(34,292),(556,323),(636,591),(13,581)
]
#area1 = [(710,429),(724,442),(775,434),(769,419)]

c = set()
#C = set()
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    
    denseCount = 0
    count += 1
    #if count % 3 != 0:
    #   continue
    frame=cv2.resize(frame,(1020,600))
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    #cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),2)
    results=model(frame,640)
    results2=model_accident(frame,416)
    #im = torch.from_numpy(frame).to(model_flood.device)
    #im = frame[None]
    #img = frame.reshape((1,frame.shape[0], frame.shape[1], frame.shape[2]))   #<------ this
    input_shape = (416, 416) # example input shape for YOLOv3
    img = cv2.resize(frame, input_shape)
    img = img.transpose((2, 0, 1)) # change HWC to CHW
    img = np.expand_dims(img, axis=0) # add batch dimension
    tensor = torch.from_numpy(img)
    device = torch.device("cuda:0")
    tensor = tensor.to(device)
    tensor = tensor.type(torch.cuda.FloatTensor)


    results3=model_flood(tensor)
    #pred_boxes = results3.pred[0][:, :4]
    bboxes = results3[0][:, :4]




    # Rescale the bounding boxes to the original image size
    original_height, original_width, _ = frame.shape
    input_height, input_width = input_shape
    bboxes[:, [0, 2]] *= 1020 / input_width
    bboxes[:, [1, 3]] *= 600 / input_height

    # Convert the tensor of bounding boxes to a list of numpy arrays
    bboxes = bboxes.cpu().numpy().tolist()

    print(bboxes)

    print(results.pandas().xyxy[0])
    print(results2.pandas().xyxy[0])

    finalResult = pd.concat([results.pandas().xyxy[0], results2.pandas().xyxy[0],results3.pandas().xyxy[0]])
    print(finalResult)
    #results=pd.concat(results2)



    points = []



    for index , row in finalResult.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        n=(row['name'])
        if 'car' in n or 'truck' in n:
            points.append([x1,y1,x2,y2])
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            #cv2.putText(frame,str(n),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        
    boxes_id = tracker.update(points) 
    #print(boxes_id)
    for box_id in boxes_id:
        x , y , w , h , idd = box_id
        cv2.rectangle(frame,(x,y),(w,h),(255,0,255),2)
        cv2.putText(frame,str(idd),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.circle(frame,(w,h),4,(0,255,0),-1)
        results = cv2.pointPolygonTest(np.array(area,np.int32),(w,h),False)
       # results1 = cv2.pointPolygonTest(np.array(area1,np.int32),(w,h),False)
        #print(results)
        if results>= 0 :
            denseCount += 1
            c.add(idd)
    #print(denseCount)    
    a = len(c)

      
   # b = len(C)
    cv2.putText(frame,'number of cars in the green region is ='+str(denseCount),(50,65),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,'Threshold ='+str(dense_threshold),(800,65),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

    if dense_threshold<denseCount:
        cv2.putText(frame,'dense traffic',(50,95),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    else:
        cv2.putText(frame,'sparse traffic',(50,95),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

  #  cv2.putText(frame,'number of cars in the yellow region is ='+str(b),(50,90),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
    cv2.imshow("FRAME",frame)
   # writer.write(frame)

    #print(a)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
#writer.release()

cv2.destroyAllWindows()

"""