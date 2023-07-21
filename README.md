# Incident Detection using YOLOv5

![Incident Detection](https://link-to-your-image.png)

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Live Stream and Video Processing](#live-stream-and-video-processing)
- [Sample Results](#sample-results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description

Incident Detection using YOLOv5 is a project aimed at automatically detecting various incidents in images and videos. The incidents that the model is trained to detect include:
- Fire
- Car accidents (car crash, car damage, car flip)
- Floods
- Dense traffic

The project utilizes the YOLOv5 object detection framework to build the detection model and a custom segmentation model for flood detection. The model is trained on labeled datasets for fire and car accidents, and a separate dataset is used for training the flood segmentation model.

The project includes components for processing images, videos, and live streams, and it utilizes websockets to send real-time incident feeds.

## Dataset

### Fire and Car Accident Detection
- The fire and car accident datasets are used to train the YOLOv5 object detection model.
- [ click here to download](https://cisasuedu-my.sharepoint.com/:u:/g/personal/abdelrahman20191700345_cis_asu_edu_eg/EbOy5N4y-zlJtUeRsqH71ccBYni8jWzdkmfqFzpaes2y3Q?e=Ueucy3)

### Flood Segmentation
- The flood segmentation dataset is used to train the custom segmentation model.
- The dataset is not publicly available on Roboflow [click here to download](https://universe.roboflow.com/le-trung-hau/flood_water)

the source dataset before annotation, from MIT License [see repository](https://github.com/ethanweber/IncidentsDataset) 

## Model Training

### YOLOv5 Object Detection Model
- The YOLOv5 model is trained on the fire and car accident datasets.
- The model weights and evaluation results are available [here](https://cisasuedu-my.sharepoint.com/:f:/g/personal/abdelrahman20191700345_cis_asu_edu_eg/EnoxUzzbAepDrPxaS3PU4Y0B55QbAEDkRl5qnBVD3az0oQ?e=de3E7D).
### Custom Segmentation Model
- The custom segmentation model for flood detection is trained on the flood segmentation dataset.
- The model weights and evaluation results are available [here](https://cisasuedu-my.sharepoint.com/:f:/g/personal/abdelrahman20191700345_cis_asu_edu_eg/EpkQget463pDlQyMFnkVyI0By16myUi1ERLiJ_4xhURfIQ?e=sKHFTi).

### YOLOv5 coco dataset weights 
- for the traffic jam class, must be detect cars and then count these cars 
- the model weights are available [here](https://cisasuedu-my.sharepoint.com/:u:/g/personal/abdelrahman20191700345_cis_asu_edu_eg/Ec0KuaAyu0RPnNsC_RfFDb0BJDZprAxwIpd20nvwvpQwpw?e=MwSwKS)

## Live Stream and Video Processing

The project includes a `detection.py` file that processes live streams, videos, and images. It utilizes the YOLOv5 model for object detection and the custom segmentation model for flood detection. The incident feeds are sent using websockets, providing real-time updates on detected incidents.

## Sample Results

Sample videos demonstrating the incident detection capabilities are available [here](https://drive.google.com/file/d/1Lp6v2fJOvIqZw6x-PuJ7vxVt9m_aX7jm/view?usp=sharing). These videos showcase the model's performance in detecting fires, car accidents, floods, and dense traffic.

## Installation

To set up the project and the required dependencies, follow these steps:
1. Clone this repository to your local machine.
2. Install the necessary packages and libraries using `pip`:

   ```bash
   pip install -r requirements.txt
3. Make sure you have the required model weights available in their respective directories 
4. Make sure you work on Gpu, run :
    ```bash
   python testGpu.py 

## Usage 
1.to run the model on video,must edit demo.py by adding your video and the weights, And you can use this command: 
  ```bash
  python demo.py