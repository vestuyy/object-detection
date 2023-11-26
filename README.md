# Contents
 - [Project Overview](#project-overview) 
 - [Prepare the environment](#1-prepare-the-environment)
 - [Setup PyCharm](#2-Setup-pycharm)
 - [Prepare the Codebase](#3-prepare-the-code-base)
 - [Define Test function](#4-define-test-function)
 - [Diagram and description](#5-diagram-and-description)
# Project Overview 

The project in discussion seems to be centered around object detection using a transformer-based model from the Hugging Face transformers library in Python. Object detection is a computer vision technique that allows us to identify and locate objects within an image or a video. This capability is crucial for many applications such as autonomous vehicles, security systems, image retrieval systems, and in industrial settings for defect detection.

Object detection models typically work by analyzing an image and identifying patterns or features that correspond to known objects. These models can be trained using large datasets of labeled images where the objects of interest are annotated with bounding boxes. Modern object detection systems often utilize deep learning, and in particular, Convolutional Neural Networks (CNNs) or Transformers to learn these features.

Transformers are a type of model that, unlike CNNs, rely on self-attention mechanisms to weigh the importance of different parts of the input data. In the context of object detection, this means that the transformer can pay more attention to the parts of the image where it believes an object is likely to be.

Here's a step-by-step breakdown of the project phases mentioned:


• Prepare the Environment:

The first step is to set up a Python environment using Anaconda, which is a popular distribution of Python for scientific computing. The given commands create a new environment with Python 3.11, and install PyTorch (a deep learning library), along with its companion libraries for vision (torchvision) and audio (torchaudio), and CUDA for GPU acceleration.
Additional libraries transformers (for accessing pretrained models and pipelines for object detection) and timm (a collection of image models) are also installed using pip.


• Setup PyCharm:

PyCharm is an Integrated Development Environment (IDE) used for Python development. Setting it up involves configuring the project to use the Anaconda environment created in the previous step and ensuring all the dependencies are recognized within the IDE.


• Prepare the Codebase:

The provided Python script serves as the foundation of the object detection project. It utilizes the transformers pipeline for object detection, loads an image from the filesystem, and performs object detection on that image.
The script defines a function draw_bounding_box that takes the coordinates of the detected object and draws a rectangular box around it, including the label and the confidence score.
After running the detection, it saves a new image with the bounding boxes drawn on top of the original image to visually represent the detected objects.


• Define Test Function:

Although not provided in the text, a test function in this context would likely involve a routine that systematically evaluates the object detection model with various test images to assess its performance. It would check if the bounding boxes are accurately placed around the objects and if the correct labels are assigned with a confidence score that reflects the model's certainty.
For the script to function correctly, the user would need to have an image named "almaty_park.jpg" in the same directory as the script and an accessible "arial.ttf" font file for text annotations. The test function would be expected to automate the process of validating the model's detection accuracy on a pre-defined set of test images.

![120116249-ea9eb900-c1a4-11eb-8265-16b1e1649867](https://github.com/vestuyy/object-detection/assets/125790973/981da44a-4b78-44be-80e4-180d36a0b14d)



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

Setting up PyCharm for a Python project involves several steps to ensure that your development environment is ready for coding. Here’s a step-by-step guide to setting up PyCharm after you've installed it on your machine:


• Launch PyCharm:

Open PyCharm either from your desktop shortcut or from the applications menu.


• Create a New Project:

On the Welcome screen, choose “Create New Project”.
In the "New Project" window, you can specify the project name and the location where it should be saved.
Make sure that the "Project Interpreter" section is set to create a new virtual environment by default. If you want to use the Anaconda environment you've created (pytorch_transformer_py3.11), you will need to configure this manually.


• Set Up Project Interpreter:

If you need to configure the Anaconda environment, click on the "Project Interpreter" dropdown.
Select "Show All".
In the next window, click on the "+" button to add a new interpreter.
Choose "Conda Environment" and then "Existing Environment".
Click on the "..." button to browse and select the Anaconda environment you have created (pytorch_transformer_py3.11).


• Install Required Packages:

Once the interpreter is set up, you can install the required packages (torch, torchvision, torchaudio, transformers, and timm) if you haven’t done so already.
Go to "File" > "Settings" (or "PyCharm" > "Preferences" on macOS) > "Project: [Your Project Name]" > "Python Interpreter".
Click on the "+" icon to add a new package. Search for the package names and install them one by one.


• Configure Project Structure:

To mark directories with special functions (like the folders for sources or tests), you can go to "File" > "Settings" > "Project: [Your Project Name]" > "Project Structure".
Select folders and mark them as "Sources", "Tests", etc., according to your project organization.


• Set Up Version Control (Optional):

If you want to use version control like Git, you can go to "VCS" > "Enable Version Control Integration".
Choose the version control system you want to use and set up the repository.


• Run Configuration:

Go to "Run" > "Edit Configurations".
Click the "+" button and select "Python".
Give a name to your configuration, select the script path to point to your main Python file, and ensure the right Python interpreter is selected.


• Code Style and Inspections:

You can customize the code style (indentations, line spacing, etc.) through "File" > "Settings" > "Editor" > "Code Style".
You can also configure code inspections by going to "File" > "Settings" > "Editor" > "Inspections" to set up how PyCharm should analyze your code for errors and standards.


• Start Coding:

With everything set up, you can start writing your code in the PyCharm editor. Use the terminal inside PyCharm or the built-in Python console to execute scripts and commands.
After you've completed these steps, your PyCharm environment should be fully configured and ready for you to begin developing your object detection project using the Python codebase you have prepared


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
PNG: 
![almaty_park](https://github.com/vestuyy/object-detection/assets/125790973/f038e378-c7db-44a0-a215-bf4e294bdba3)
Result :
![almaty_park jpg_boxes](https://github.com/vestuyy/object-detection/assets/125790973/d2ad29d6-c7ac-4c64-b8d9-1c8f5f163815)



# 5. Diagram and description 


Project Workflow Description:


• Environment Preparation:

A new conda environment named pytorch_transformer_py3.11 is created and activated, using Python 3.11.
Key libraries for deep learning and transformers, including PyTorch, torchvision, torchaudio, and CUDA support, are installed.
Additional packages such as transformers and timm are installed via pip.


• PyCharm Setup:

PyCharm, an Integrated Development Environment (IDE), is set up for Python development. This typically involves configuring the interpreter, creating a project, and possibly setting up version control.


• Codebase Overview:

The script imports necessary libraries (transformers, PIL) and defines functions.
A function print_hi is defined for initial greetings.
A function draw_bounding_box is created to draw bounding boxes with labels around detected objects in an image.
The main execution block loads an image, initializes an object detection model from the Hugging Face transformers library, processes the image to detect objects, draws bounding boxes with labels around detected objects, and saves the resultant image.

Flow Diagram:
```python
+---------------------+      +----------------------+      +--------------------------+
|    Create and       |      |      Activate       |      |    Install Libraries     |
| Activate Environment| ---> |   Conda Environment | ---> | (PyTorch, transformers)  |
+---------------------+      +----------------------+      +--------------------------+
                                       |
                                       V
                              +-------------------+      +------------------------+
                              |   Setup PyCharm   | ---> |   Write/Load Python    |
                              |   for Development |      |      Script            |
                              +-------------------+      +------------------------+
                                                                   |
                                                                   V
                              +--------------------------+      +-----------------------------------+
                              |   Initialize Object      |      |   Process Image and Detect        |
                              |  Detection Pipeline      | ---> |   Objects with Bounding Boxes     |
                              +--------------------------+      +-----------------------------------+
                                                                   |
                                                                   V
                                                          +-----------------------------+
                                                          |   Save Image with Drawn     |
                                                          |   Bounding Boxes            |
                                                          +-----------------------------+

```
This flowchart demonstrates the sequential steps taken in the project, from setting up the development environment to running the object detection script and saving the output. The actual complexity of the diagram can be much greater, depending on the depth of detail required (such as error handling, testing procedures, etc.). In a software diagramming tool, this workflow would typically be represented with more formal symbols and connections, but the above ASCII diagram provides a simplified visual representation of the process.

#6. Technical Details

How Transformers Work
Transformers are a groundbreaking architecture in the field of natural language processing, introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. Unlike previous models that relied on sequential data processing (like RNNs and LSTMs), transformers use a mechanism called self-attention to process input data in parallel, leading to significant improvements in efficiency and performance.

Key Components of Transformers:

• Encoder and Decoder Blocks: The transformer model consists of encoder and decoder blocks. The encoder processes the input data, and the decoder generates the output. In tasks like object detection, often only the encoder part is used.

• Self-Attention Mechanism: This is the core of the transformer. It allows the model to weigh the importance of different parts of the input data. For example, in a sentence, the model can focus more on relevant words while processing a particular word.

• Positional Encoding: Since transformers do not process data sequentially, they need a way to understand the order of the input. Positional encoding is added to the input embeddings to provide the model with information about the position of each element in the sequence.

• Multi-Head Attention: This involves running the self-attention mechanism multiple times in parallel. The independent attention outputs are then combined and processed, allowing the model to focus on different positions and capture a broader context.
 
• Feed-Forward Networks: Each encoder and decoder block contains a feed-forward network that applies linear transformations to the output of the attention layer, followed by a non-linear activation function.

• Layer Normalization and Residual Connections: These components help in stabilizing the learning process and allow for deeper networks by preventing the vanishing gradient problem.

Object Detection Pipeline in Transformers
In the context of object detection, transformers are used to process images and identify objects within them. The process typically involves the following steps:

•I mage Preprocessing: The input image is preprocessed into a format suitable for the transformer model, often involving resizing and normalization.

• Feature Extraction: In some transformer-based models, features from the image are first extracted using a CNN. These features are then passed to the transformer.
 
• Applying Transformers: The transformer processes the image (or extracted features) using its self-attention mechanism. This allows the model to focus on relevant parts of the image for object detection.

• Bounding Box Prediction: The transformer outputs predictions, which include the coordinates for bounding boxes around detected objects, along with classification labels and confidence scores.

• Post-Processing: This step may involve filtering overlapping boxes using methods like Non-Maximum Suppression (NMS) and thresholding the confidence scores to improve the precision of object detection.
