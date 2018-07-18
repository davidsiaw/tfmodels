######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'input.png'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 19

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_to_id_map = {}
for x in categories:
    label_to_id_map[x['name']] = x['id']

label_to_id_map['hanamaru'] = label_to_id_map['maru']

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=detection_graph,config=config)

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

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)


# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

sboxes = np.squeeze(boxes)
sclasses = np.squeeze(classes).astype(np.int32)
sscores = np.squeeze(scores)

for i in range(sboxes.shape[0]):
    if sscores[i] > 0.70:
        ymin, xmin, ymax, xmax = sboxes[i]
        #print(category_index[sclasses[i]]['name'])
        im = Image.open(PATH_TO_IMAGE)
        print('{}%'.format(int(100*sscores[i])), xmin, xmax, ymin, ymax )
        width, height = im.size
        x0 = width * xmin
        y0 = height * ymin
        x1 = width * xmax
        y1 = height * ymax
        im = im.crop( (x0, y0, x1, y1) )
        file_name = str(i) + '.png'
        model_file = 'c:/cuda/test/tf_files/retrained_graph.pb'
        label_file = "c:/cuda/test/tf_files/retrained_labels.txt"
        input_height = 224
        input_width = 224
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder"
        output_layer = "final_result"
        im.save(file_name)
        graph = load_graph(model_file)
        t = read_tensor_from_image_file(
          file_name,
          input_height=input_height,
          input_width=input_width,
          input_mean=input_mean,
          input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config,graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        labelid = label_to_id_map[ labels[top_k[0]] ]
        print(labels[top_k[0]], '(',category_index[sclasses[i]]['name'],')', results[top_k[0]])
        sclasses[i] = labelid
        sscores[i] = (results[top_k[0]] + sscores[i]) / 2

class_name_to_color_map = {
    'honoka': 'DarkOrange',
    'umi': 'Blue',
    'kotori': 'White',
    'rin': 'Yellow',
    'hanayo': 'LimeGreen',
    'maki': 'Crimson',
    'nico': 'HotPink',
    'eli': 'LightBlue',
    'nozomi': 'MediumOrchid',

    'chika': 'Orange',
    'you': 'DeepSkyBlue',
    'riko': 'LightPink',
    'ruby': 'DeepPink',
    'maru': 'Gold',
    'yohane': 'GainsBoro',
    'dia': 'Red',
    'kanan': 'MediumSpringGreen',
    'mari': 'BlueViolet',
}

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    sboxes,
    sclasses,
    sscores,
    category_index,
    use_normalized_coordinates=True,
    class_name_to_color_map=class_name_to_color_map,
    line_thickness=3,
    min_score_thresh=0.75)

cv2.imwrite("output.png", image)
#All the results have been drawn on image. Now display the image.
#cv2.imshow('Object detector', image)

#Press any key to close the image
#cv2.waitKey(0)

#Clean up
cv2.destroyAllWindows()
