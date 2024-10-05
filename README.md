# Lava Tubes Detection Project

## Overview
This project focuses on the detection of lava tubes in satellite imagery using machine learning techniques. Lava tubes are natural conduits formed by flowing lava, and their detection is crucial for understanding volcanic activity, assessing hazards, and exploring potential habitats for future extraterrestrial missions.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Project Description
The goal of this project is to develop a robust model capable of accurately identifying lava tubes in various types of imagery, including satellite and aerial photographs. Using a combination of data augmentation techniques and a YOLO (You Only Look Once) architecture, we aim to enhance the model's precision and recall rates.

### Objectives:
- To accurately detect and classify lava tubes in satellite images.
- To minimize false positives and false negatives in model predictions.
- To visualize the detection results effectively.

## Dataset
The dataset used in this project consists of labeled images that contain lava tubes and other geological features. The images were sourced from [insert source, e.g., public datasets, satellite imagery], and the annotations include bounding boxes around identified lava tubes.

- **Number of Images**: [insert number]
- **Classes**: 
  - Lava Tube 1
  - Lava Tube 2
  - Random

## Installation
To run the project, ensure you have the following prerequisites installed:

1. Python 3.x
2. Required libraries (listed in `requirements.txt`):
   ```bash
   pip install -r requirements.txt





   1. Setup Environment
Import Libraries and Initialize Roboflow
python
Copy code
from roboflow import Roboflow

# Initialize the Roboflow API with your API key
rf = Roboflow(api_key="owkxxTTrJVyG8LYQMgXv")

# Access the specific project and version of the dataset
project = rf.workspace("aerospace-iw0lh").project("hackathon-gux9o")
version = project.version(3)  # Specify the version to download

# Download the dataset formatted for YOLOv9
dataset = version.download("yolov9")
2. Train the Model
Change Directory and Train the Model
python
Copy code
%cd {HOME}/yolov9  # Change directory to where YOLOv9 is located

# Execute the training script with specified parameters
!python train.py \
--batch 16 \  # Number of images to process at once
--epochs 50 \  # Number of training iterations
--img 640 \  # Image size for training
--device 0 \  # GPU device to use (0 for the first GPU)
--min-items 0 \  # Minimum items per class required for training
--close-mosaic 15 \  # Close mosaic augmentation after 15 epochs
--data {dataset.location}/data.yaml \  # Path to dataset configuration
--weights {HOME}/weights/gelan-c.pt \  # Path to initial weights
--cfg models/detect/gelan-c.yaml \  # Model configuration file
--hyp hyp.scratch-high.yaml  # Hyperparameter configuration file
View Training Results
python
Copy code
# List the contents of the training results directory
!ls {HOME}/yolov9/runs/train/exp/

# Display the training results image (e.g., loss curves)
from IPython.display import Image
Image(filename=f"{HOME}/yolov9/runs/train/exp/results.png", width=1000)

# Display the confusion matrix image
Image(filename=f"{HOME}/yolov9/runs/train/exp/confusion_matrix.png", width=1000)
3. Validate the Model
Validate the Model with Test Data
python
Copy code
%cd {HOME}/yolov9  # Change back to the YOLOv9 directory

# Execute the validation script with specified parameters
!python val.py \
--img 640 \  # Image size for validation
--batch 32 \  # Number of images to process at once during validation
--conf 0.001 \  # Confidence threshold for detections
--iou 0.7 \  # Intersection over Union threshold
--device 0 \  # GPU device to use (0 for the first GPU)
--data {dataset.location}/data.yaml \  # Path to dataset configuration
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt  # Path to the best weights from training
4. Make Predictions
Run Object Detection on Test Images
python
Copy code
# Execute the detection script on test images
!python detect.py \
--img 1280 \  # Image size for inference
--conf 0.1 \  # Confidence threshold for detections
--device 0 \  # GPU device to use (0 for the first GPU)
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \  # Path to the best weights
--source {dataset.location}/test/images  # Path to the directory with test images

# Display the first two detected images
import glob
from IPython.display import Image, display

# Loop through detected images and display them
for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp3/*.jpg')[:2]:
      display(Image(filename=image_path, width=600))
5. Install Necessary Packages
bash
Copy code
# Install required packages for inference and supervision
!pip install inference
!pip install supervision
6. Deployment
Deploy the Model
python
Copy code
import cv2
import random
import getpass
import supervision as sv
from inference import get_model

%matplotlib inline  # Enable inline plotting for Jupyter notebooks

# Deploy the model using Roboflow
version.deploy(model_type="yolov9", model_path=f"{HOME}/yolov9/runs/train/exp")

# Get the API key for accessing the model
ROBOFLOW_API_KEY = getpass.getpass()

# Load the model for inference from Roboflow
model = get_model(model_id="hackathon-gux9o/4", api_key="owkxxTTrJVyG8LYQMgXv")

# Prepare the images for inference from the test directory
image_paths = sv.list_files_with_extensions(
    directory=f"{dataset.location}/test/images",
    extensions=['png', 'jpg', 'jpeg']
)
image_path = random.choice(image_paths)  # Select a random image for inference
image = cv2.imread(image_path)  # Read the selected image

# Perform inference on the selected image
result = model.infer(image, confidence=0.1)[0]  # Run inference with specified confidence
detections = sv.Detections.from_inference(result)  # Convert results to detections format
