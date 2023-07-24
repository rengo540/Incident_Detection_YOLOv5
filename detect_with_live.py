import base64
from datetime import datetime
import os
import cv2
import torch
import pytz

import numpy as np
import time
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



import asyncio
import websockets







#to determine points in image 
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        


#read weights 
device = select_device('')
imgsz=(640, 640)  
model = DetectMultiBackend('yolov5m.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Carnames, pt = model.stride, model.names, model.pt


model_flood = DetectMultiBackend('runs/train-seg/modelFloods53/weights/last.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Floodnames, pt = model_flood.stride, model_flood.names, model_flood.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

model_accident = DetectMultiBackend('afterSeminar/modelVer6/weights/last.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Accidentnames, pt = model_accident.stride, model_accident.names, model_accident.pt



#parameters 
writer= cv2.VideoWriter('output/live.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1320,600))
#exampleClips/CCTVCARCRASHES.mp4
#https://192.168.1.12:8080/video
#https://www.youtube.com/watch?v=5_XSYlAfJZM&ab_channel=BradPhillips
#https://www.youtube.com/watch?v=RQA5RcIZlAM&ab_channel=%E3%80%90LIVE%E3%80%91%E6%96%B0%E5%AE%BF%E5%A4%A7%E3%82%AC%E3%83%BC%E3%83%89%E4%BA%A4%E5%B7%AE%E7%82%B9TokyoShinjukuLiveCh
source = str('https://192.168.1.12:8080/video')
area = [(0,0),(0,0),(0,0),(0,0)]
dense_threshold=6
global_url ="wss://giant-clever-corn.glitch.me/"

conf_thres=0.5  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
bs = 1  # batch_size


is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
screenshot = source.lower().startswith('screen')
if is_url and is_file:
    source = check_file(source)  # download



#download source 
if webcam:
    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=4)
    bs = len(dataset)
elif screenshot:
    dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=4)







def is_bbox_intersected(bbox1, bbox2):

    x1, w1, y1, h1 = bbox1
    x2, w2, y2, h2 = bbox2

    # Calculate the coordinates of the corners of the bounding boxes
    bbox1_corners = [(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)]
    bbox2_corners = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]

    # Check if any corner of bbox2 is inside bbox1
    for corner in bbox2_corners:
        if corner[0] >= x1 and corner[0] <= x1 + w1 and corner[1] >= y1 and corner[1] <= y1 + h1:
            return True

    # Check if any corner of bbox1 is inside bbox2
    for corner in bbox1_corners:
        if corner[0] >= x2 and corner[0] <= x2 + w2 and corner[1] >= y2 and corner[1] <= y2 + h2:
            return True

    return False











 # Run inference



model_flood.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model_accident.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

seen, windows, dt = 0, [], (Profile(), Profile(), Profile())



async def send_data():
    try:
        async with websockets.connect(global_url) as websocket:

            for path, im, im0s, vid_cap, s in dataset:
                cv2.namedWindow('img')
                cv2.setMouseCallback('img', POINTS)


                incident_report = { 
                    'Is_Incident':0,
                    'incidents':{"fire":0 , "car damage":0, "car crash":0 , "car flip":0, "flooding":0},
                    'dense_traffic' : 0,
                    'frame': 'None',
                    'time':time.time(),
                    'camera_id':1,
                    'camera_position':[37.41941723838883, -122.08535405545874]

                    }
                df = pd.DataFrame(columns=['x1', 'x2', 'y1', 'y2', 'confidence', 'class'])
                dense_count=0

                
                #preprocessing
                with dt[0]:
                    im = torch.from_numpy(im).to(model_flood.device)
                    im = im.half() if model_flood.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim                
 
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
                
                points=[]
                # Car predictions
                for i, det in enumerate(car_pred):  # per image
                    #seen += 1
                    

                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)


                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            x1 = int(xyxy[0].item())
                            y1 = int(xyxy[1].item())
                            x2 = int(xyxy[2].item())
                            y2 = int(xyxy[3].item())

                            #class_index = cls
                            object_name = Carnames[int(cls)]
                            new_row = {'x1': x1, 'x2': x2,'y1':y1,'y2':y2,'confidence':conf,'class':object_name}
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)                        
                            


                            points.append([x1,y1,x2,y2])



                            #count in region of interest 
                            results = cv2.pointPolygonTest(np.array(area,np.int32), (x2,y2) ,False)
                            if results>=0:
                                dense_count += 1




                # accident predictions
                for i, det in enumerate(accident_pred):  # per image
                    #seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    
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
                            #names: ['car damage' ,'fire','car flip','car','car crash']

                            object_name = Accidentnames[int(cls)]
                            if not(incident_report['incidents'][object_name]):
                                incident_report['incidents'][object_name]=1
                                incident_report['Is_Incident']=1
                            
                            new_row = {'x1': x1, 'x2': x2,'y1':y1,'y2':y2,'confidence':conf,'class':object_name}
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)                        
                             

                                


                # floods predictions
                for i, det in enumerate(floods_pred):  # per image
                
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    
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
                                if not(incident_report['incidents'][object_name]):
                                    incident_report['incidents'][object_name]=1
                                    incident_report['Is_Incident']=1

                                new_row = {'x1': x1, 'x2': x2,'y1':y1,'y2':y2,'confidence':conf,'class':object_name}
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)                        
                            
                            


                
                
                for index, row in df.iterrows():
                    x1 = row['x1']
                    x2 = row['x2']
                    y1 = row['y1']
                    y2 = row['y2']
                    confidence = row['confidence']
                    class_name = row['class']

                    rec_color = (0,0,255)
                    if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                        rec_color = (255,0,0) 
                        cv2.circle(im0,(x2,y2),4,(0,255,0),-1)

                    cv2.rectangle(im0,(x1,y1),(x2,y2),rec_color,3)
                    cv2.putText(im0,str(class_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,rec_color,2) 





                top, bottom, left, right = 0, 0, 0, 300
                border_color = [255, 255, 255]  # White color
                # Add the border
                im0=cv2.resize(im0,(1020,600))
                im0s_with_border = cv2.copyMakeBorder(im0, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
                  
                if dense_threshold < dense_count:
                    if not(incident_report['dense_traffic']):
                        incident_report['dense_traffic']=1
                        incident_report['Is_Incident']=1
    
                    cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,0,255),thickness=2,lineType=cv2.LINE_AA)
                    cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                    cv2.putText(im0s_with_border,'dense traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                else:
                    cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,255,0),thickness=2,lineType=cv2.LINE_AA)
                    cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
                    cv2.putText(im0s_with_border,'sparse traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
                
                

                if(incident_report['Is_Incident']):
                    
                    im0=cv2.resize(im0,(150,150))
                    ret, buffer = cv2.imencode('.jpg', im0)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    incident_report['frame']=jpg_as_text


                # Send the data to the WebSocket server
                report = str(incident_report)
                await websocket.send(report)
                await asyncio.sleep(0.001)

                cv2.imshow("img",im0s_with_border)
                #writer.write(im0s_with_border)

                if cv2.waitKey(1)&0xFF==27:
                    break
            writer.release()

            cv2.destroyAllWindows()
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed: {e}")
        # Handle the closed connection, such as reconnecting or stopping the loop
        await asyncio.sleep(1)  # Delay before reconnecting (adjust as needed)
        await send_data()


asyncio.get_event_loop().run_until_complete(send_data())
