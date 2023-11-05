# Contents
 - [Project Overview](#project-overview) 
 - [Prepare the environment](#1-prepare-the-environment)
 - [Setup PyCharm](#2-Setup-pycharm)
 - [Prepare the Codebase](#3-prepare-the-code-base)
 - [Define Test function](#4-define-test-function)
 
# Project Overview 

Write about project overview. Explain what is a object detection and how it works.  



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
