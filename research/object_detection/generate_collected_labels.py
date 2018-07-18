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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import xml.etree.cElementTree as ET
from xml.dom import minidom
from PIL import Image
import hashlib


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

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

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 21

## Load the label map.
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


model_file = 'c:/cuda/test/tf_files/retrained_graph.pb'
label_file = "c:/cuda/test/tf_files/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"
graph = load_graph(model_file)

input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
classifier_sess = tf.Session(config=config,graph=graph)


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

# Get all files

class_name_to_color_map = {
    'honoka': 'DarkOrange',
    'umi': 'Blue',
    'kotori': 'White',
    'rin': 'Yellow',
    'hanayo': 'LimeGreen',
    'maki': 'Crimson',
    'nico': 'HotPink',
    'eli': 'Cyan',
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

    'sarah': 'LightSkyBlue',
    'leah': 'Snow',
}

relevant_path = "collected"
included_extensions = ['jpg', 'png']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

for filename in file_names:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread("collected/"+filename)

    r = 600.0 / image.shape[1]
    dim = (600, int(image.shape[0] * r))
     
    # perform the actual resizing of the image and show it
    image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    image_expanded = np.expand_dims(image, axis=0)

    overwrite = True

    dest_folder = 'C:/cuda/test/models/research/object_detection/labelimg/collected/'

    cls_dest_folder = 'C:/cuda/test/models/research/object_detection/cls/'

    xmlfilename = dest_folder+filename.replace(".png","").replace(".jpg","")+".xml"

    try:
        existing_tree=ET.parse(xmlfilename)
        existing_root=existing_tree.getroot()
        if existing_root.attrib['verified'] == 'yes':
            overwrite = False
    except:
        pass

    if overwrite:
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        dest = dest_folder+filename

        annotation = ET.Element("annotation", verified="no")
        ET.SubElement(annotation, "folder").text = "collected"
        ET.SubElement(annotation, "filename").text = filename
        ET.SubElement(annotation, "path").text = dest.replace("/", "\\")
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image.shape[1])
        ET.SubElement(size, "height").text = str(image.shape[0])
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        sboxes = np.squeeze(boxes)
        sclasses = np.squeeze(classes).astype(np.int32)
        sscores = np.squeeze(scores)

        for i in range(sboxes.shape[0]):
            if sscores[i] > 0.65:
                ymin, xmin, ymax, xmax = sboxes[i]
                x0 = image.shape[1] * xmin
                y0 = image.shape[0] * ymin
                x1 = image.shape[1] * xmax
                y1 = image.shape[0] * ymax

                im = Image.open('collected/'+filename)
                im = im.crop( (x0, y0, x1, y1) )
                file_name = 'bb' + str(i) + '.png'
                im.save(file_name)
                t = read_tensor_from_image_file(
                  file_name,
                  input_height=input_height,
                  input_width=input_width,
                  input_mean=input_mean,
                  input_std=input_std)

                results = classifier_sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)
                labelid = label_to_id_map[ labels[top_k[0]] ]
                print(labels[top_k[0]], '(',category_index[sclasses[i]]['name'],')', results[top_k[0]])
                sclasses[i] = labelid
                sscores[i] = (results[top_k[0]] + sscores[i]) / 2

                class_folder = cls_dest_folder+category_index[sclasses[i]]['name']+'/'

                if not os.path.exists(class_folder):
                  os.makedirs(class_folder)

                openedFile = open(file_name, 'rb')
                readFile = openedFile.read()
                sha1Hash = hashlib.sha1(readFile)
                sha1Hashed = sha1Hash.hexdigest()

                im.save(class_folder+sha1Hashed+'.png')

                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = category_index[sclasses[i]]['name']
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                if sscores[i] > 0.9:
                    ET.SubElement(obj, "difficult").text = "0"
                else:
                    ET.SubElement(obj, "difficult").text = "1"
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(x0))
                ET.SubElement(bndbox, "ymin").text = str(int(y0))
                ET.SubElement(bndbox, "xmax").text = str(int(x1))
                ET.SubElement(bndbox, "ymax").text = str(int(y1))


        tree = ET.ElementTree(annotation)

        xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="\t")

        with open(xmlfilename, "w") as f:
            f.write(xmlstr)

        cv2.imwrite(dest, image)

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_resized,
            sboxes,
            sclasses,
            sscores,
            category_index,
            use_normalized_coordinates=True,
            class_name_to_color_map=class_name_to_color_map,
            line_thickness=3,
            min_score_thresh=0.75)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', image_resized)
    else:
        print("skipping " + dest_folder + filename + " because verified")

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

