######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import collections
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
STANDARD_COLORS = [
     'Lime','aqua', 'pink'

]

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def test(
    image,
    boxes,
    classes,
    scores,
    max_boxes_to_draw=20,
    min_score_thresh=.5,


    ):

  # Create scatter and decide location
  # that correspond to the same location.

  box_to_color_map = collections.defaultdict(str)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if scores is None:
        print(scores)
        box_to_color_map[box] = 'black'
      else:
        box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.


  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    Xc=(xmin+xmax)/2
    Yc=(ymin+ymax)/2

    return Xc,Yc,color



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

d={1:0,2:0,3:0,4:0,5:0,6:0}
# Initialize webcam feed
video = cv2.VideoCapture(1) #set where webacm
ret = video.set(4,1280)
ret = video.set(5,720)
plt.figure()
count=0
XcTest=0
YcTest=0
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)

    z=test(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        min_score_thresh=0.85)

    if z is None:
        pass
    else:
        x,y,color=z

        if ((x- XcTest) > 0.2 or (x- XcTest) < -0.2):
            XcTest=x
            YcTest=y
            print(x, y)
            plt.scatter(x, y, c=color, alpha=0.5)
            if (x < 0.35):
                if (STANDARD_COLORS[1] == color):
                    bar_count= 1
                else:
                    bar_count = 2

            elif (x > 0.35 and x < 0.7):
                if (STANDARD_COLORS[1] == color):
                    bar_count = 3
                else:
                    bar_count = 4
            else:
                if (STANDARD_COLORS[1] == color):
                    bar_count = 5
                else:
                    bar_count = 6
            d[bar_count] += 1





    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
print(d)
plt.xlim((0.1,1.0))
plt.ylim((1.0,0.1))
plt.figure()
plt_x=np.arange(len(d))+1
plt_y=np.array([d[1],d[2],d[3],d[4],d[5],d[6]])
plt.bar(plt_x,plt_y,color=['aqua', 'pink'])
labels=['MaleR','FemaleR','MaleC','FemaleC','MaleL','FemaleL']
plt.xticks(plt_x,labels)
plt.show()
#Clean up
video.release()
cv2.destroyAllWindows()
