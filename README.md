# Contents
 - [Project Overview](#project-overview) 
 - [Prepare the environment](#1-prepare-the-environment)
 - [Setup PyCharm](#2-Setup-pycharm)
 - [Prepare the Codebase](#3-prepare-the-code-base)
 - [Define Test function](#4-define-test-function)
 
# Project Overview 

The project in discussion seems to be centered around object detection using a transformer-based model from the Hugging Face transformers library in Python. Object detection is a computer vision technique that allows us to identify and locate objects within an image or a video. This capability is crucial for many applications such as autonomous vehicles, security systems, image retrieval systems, and in industrial settings for defect detection.

Object detection models typically work by analyzing an image and identifying patterns or features that correspond to known objects. These models can be trained using large datasets of labeled images where the objects of interest are annotated with bounding boxes. Modern object detection systems often utilize deep learning, and in particular, Convolutional Neural Networks (CNNs) or Transformers to learn these features.

Transformers are a type of model that, unlike CNNs, rely on self-attention mechanisms to weigh the importance of different parts of the input data. In the context of object detection, this means that the transformer can pay more attention to the parts of the image where it believes an object is likely to be.

Here's a step-by-step breakdown of the project phases mentioned:

[1] Prepare the Environment:

The first step is to set up a Python environment using Anaconda, which is a popular distribution of Python for scientific computing. The given commands create a new environment with Python 3.11, and install PyTorch (a deep learning library), along with its companion libraries for vision (torchvision) and audio (torchaudio), and CUDA for GPU acceleration.
Additional libraries transformers (for accessing pretrained models and pipelines for object detection) and timm (a collection of image models) are also installed using pip.

[2] Setup PyCharm:

PyCharm is an Integrated Development Environment (IDE) used for Python development. Setting it up involves configuring the project to use the Anaconda environment created in the previous step and ensuring all the dependencies are recognized within the IDE.

[3] Prepare the Codebase:

The provided Python script serves as the foundation of the object detection project. It utilizes the transformers pipeline for object detection, loads an image from the filesystem, and performs object detection on that image.
The script defines a function draw_bounding_box that takes the coordinates of the detected object and draws a rectangular box around it, including the label and the confidence score.
After running the detection, it saves a new image with the bounding boxes drawn on top of the original image to visually represent the detected objects.

[4] Define Test Function:

Although not provided in the text, a test function in this context would likely involve a routine that systematically evaluates the object detection model with various test images to assess its performance. It would check if the bounding boxes are accurately placed around the objects and if the correct labels are assigned with a confidence score that reflects the model's certainty.
For the script to function correctly, the user would need to have an image named "almaty_park.jpg" in the same directory as the script and an accessible "arial.ttf" font file for text annotations. The test function would be expected to automate the process of validating the model's detection accuracy on a pre-defined set of test images.


# 1. Prepare the environment  


### 1. Open Anaconda prompt and exectute the following commands to setup the environment
 
```python
conda create --name pytorch_transformer_py3.11 python==3.11
conda activate pytorch_transformer_py3.11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers

pip install timm

```

### 2. Setup PyCharm 

Here is how to setup PyCharm 


# 3. Prepare the Codebase


```python

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Draw bounding box definition
def draw_bounding_box(im, score, label, xmin, ymin, xmax, ymax, index, num_boxes):
	""" Draw a bounding box. """

	print(f"Drawing bounding box {index} of {num_boxes}...")

	# Draw the actual bounding box
	im_with_rectangle = ImageDraw.Draw(im)
	im_with_rectangle.rounded_rectangle((xmin, ymin, xmax, ymax), outline = "red", width = 5, radius = 10)

	# Draw the label
	im_with_rectangle.text((xmin+35, ymin-25), label, fill="white", stroke_fill = "red", font = font)

	# Return the intermediate result
	return im

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	print_hi('Hello Object Detection Using Transformer')
	name_of_the_file = "almaty_park.jpg"

	# Load font
	font = ImageFont.truetype("arial.ttf", 40)

	# Initialize the object detection pipeline
	object_detector = pipeline("object-detection")

	# # Open the image
	with Image.open(name_of_the_file) as im:

		# Perform object detection
		bounding_boxes = object_detector(im)

		# Iteration elements
		num_boxes = len(bounding_boxes)
		index = 0

		# Draw bounding box for each result
		for bounding_box in bounding_boxes:
			# Get actual box
			box = bounding_box["box"]

			# Draw the bounding box
			im = draw_bounding_box(im, bounding_box["score"], bounding_box["label"], \
								   box["xmin"], box["ymin"], box["xmax"], box["ymax"], index, num_boxes)

			# Increase index by one
			index += 1

		# Save image
		im.save(name_of_the_file+"_boxes.jpg")

		# Done
		print("Done!")









```

# 4. Define Test function
