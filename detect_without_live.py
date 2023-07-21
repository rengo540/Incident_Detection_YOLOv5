import base64
import os
import cv2
import torch
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








def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        



device = select_device('')
imgsz=(640, 640)  
model = DetectMultiBackend('yolov5m.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Carnames, pt = model.stride, model.names, model.pt


model_flood = DetectMultiBackend('runs/train-seg/modelFloods53/weights/last.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Floodnames, pt = model_flood.stride, model_flood.names, model_flood.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

model_accident = DetectMultiBackend('afterSeminar/modelVer6/weights/last.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, Accidentnames, pt = model_accident.stride, model_accident.names, model_accident.pt


writer= cv2.VideoWriter('output/crashOutput/dense3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1320,600))


source = str('exampleClips/dense3.mp4')
conf_thres=0.5  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
bs = 1  # batch_size
area = [(964,567),(911,188),(346,167),(11,514)]
dense_threshold=15
#tracker = Tracker()

dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=2)













 # Run inference
model_flood.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model_accident.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

global_url ="wss://giant-clever-corn.glitch.me/"
async def send_data():
    async with websockets.connect(global_url) as websocket:
    
        for path, im, im0s, vid_cap, s in dataset:
            cv2.namedWindow('img')
            cv2.setMouseCallback('img', POINTS)


            incident_report = { 
                'Is_Incident':False,
                'incidents':{"fire":False , "car damage":False, "car crash":False , "car flip":False, "flooding":False},
                'dense_traffic' : False,
                'frame': None,
                'time':time.time(),
                'camera_position':[30.127203, 31.299466]

                }

            
          
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
                #seen += 1
                
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

                        
                        cv2.rectangle(im0s,(x1,y1),(x2,y2),(255,0,0),3)
                        cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)         
                        cv2.circle(im0s,(x2,y2),4,(0,255,0),-1)
                        results = cv2.pointPolygonTest(np.array(area,np.int32), (x2,y2) ,False)
                        if results>=0:
                            dense_count += 1




        # accident predictions
            
            for i, det in enumerate(accident_pred):  # per image
                #seen += 1
                
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
                        #names: ['car damage' ,'fire','car flip','car','car crash']

                        object_name = Accidentnames[int(cls)]
                        if not(incident_report['incidents'][object_name]):
                            incident_report['incidents'][object_name]=True
                            incident_report['Is_Incident']=True
                        
                            
                        
                        #points.append([x1,x2,x2,y2,con,n])
                        cv2.rectangle(im0s,(x1,y1),(x2,y2),(0,0,255),3)
                        cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)



            # floods predictions
            for i, det in enumerate(floods_pred):  # per image
                #seen += 1
            
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
                            if not(incident_report['incidents'][object_name]):
                                incident_report['incidents'][object_name]=True
                                incident_report['Is_Incident']=True


                        
                            cv2.rectangle(im0s,(x1,y1),(x2,y2),(0,0,255),3)
                            cv2.putText(im0s,str(object_name),(x1,y1-7),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

            top, bottom, left, right = 0, 0, 0, 300
            border_color = [255, 255, 255]  # White color
            # Add the border
            im0s_with_border = cv2.copyMakeBorder(im0s, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

        
            cv2.putText(im0s_with_border,'INCIDENT TYPE',(1050,160),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)

            

            cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
            cv2.putText(im0s_with_border,'Dense Threshold ='+str(dense_threshold),(1050,120),cv2.FONT_ITALIC,0.5,(0,0,0),1)                         
            if dense_threshold < dense_count:
                if not(incident_report['dense_traffic']):
                    incident_report['dense_traffic']=True
                    incident_report['Is_Incident']=True

                        
                cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,0,255),thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                cv2.putText(im0s_with_border,'dense traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            else:
                cv2.polylines(im0s_with_border,[np.array(area,np.int32)],True,(0,255,0),thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0s_with_border,str(dense_count)+' Vehicle',(1050,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
                cv2.putText(im0s_with_border,'sparse traffic',(1050,95),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
            
            
            #names: ['car damage' ,'fire','car flip','car','car crash']

            if(incident_report['Is_Incident']):
                 ret, buffer = cv2.imencode('.jpg', frame)
                 jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                 incident_report['frame']=jpg_as_text


            # Send the data to the WebSocket server
            report = str(incident_report)
            print(report)
            await websocket.send(report)
            await asyncio.sleep(0.001)

            cv2.imshow("img",im0s_with_border)
            #writer.write(im0s_with_border)

            if cv2.waitKey(1)&0xFF==27:
                break
        writer.release()

        cv2.destroyAllWindows()


asyncio.get_event_loop().run_until_complete(send_data())
