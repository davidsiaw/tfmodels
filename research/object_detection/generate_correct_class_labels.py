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

SOURCE_DIR="C:/cuda/test/models/research/object_detection/labelimg/collected.1"
DEST_DIR="C:/cuda/test/models/research/object_detection/correctcls"
included_extensions = ['xml']

file_names = [fn for fn in os.listdir(SOURCE_DIR)
              if any(fn.endswith(ext) for ext in included_extensions)]

for xmlfile in file_names:
  tree = ET.parse(SOURCE_DIR+"/"+xmlfile)
  print(xmlfile)
  root = tree.getroot()
  if 'verified' in root.attrib and root.attrib['verified'] == 'yes':
    imgpath = root.find('path').text
    width = float(root.find('size').find('width').text)
    height = float(root.find('size').find('height').text)
    items = []
    for obj in root.findall('object'):
      item = {}
      item['name'] = obj.find('name').text
      item['x0'] = float(obj.find('bndbox').find('xmin').text)
      item['x1'] = float(obj.find('bndbox').find('xmax').text)
      item['y0'] = float(obj.find('bndbox').find('ymin').text)
      item['y1'] = float(obj.find('bndbox').find('ymax').text)
      items.append(item)

      im = Image.open(imgpath)
      im = im.crop( (item['x0'], item['y0'], item['x1'], item['y1']) )
      im.save('tmp.png')

      openedFile = open('tmp.png', 'rb')
      readFile = openedFile.read()
      sha1Hash = hashlib.sha1(readFile)
      sha1Hashed = sha1Hash.hexdigest()

      class_folder = DEST_DIR+"/"+item['name']

      if not os.path.exists(class_folder):
        os.makedirs(class_folder)

      im.save(class_folder+"/"+sha1Hashed+'.png')
  print(str(len(items))+" objects")


