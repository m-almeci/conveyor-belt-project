# Imports
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

# Get data path
"""
myFile = 'flower_photos.tgz'
fullPath = os.path.abspath("./" + myFile)
image_path = tf.keras.utils.get_file(myFile, 'file://'+fullPath)

"""
"""
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://www.storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos') 
image_path = tf.keras.utils.get_file(
      'componentes.tgz',
      'https://www.github.com/m-almeci/conveyor_belt/raw/main/training/componentes.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'componentes')


'https://www.drive.google.com/uc?export=download&id=1f3m3Vmn-fEJ749lO_F0R_N91BOnoXgVJ'
componentes.tgz
'https://www.drive.google.com/uc?export=download&id=18qnEEom4vrVy2j4_rQwU3O2viivGZRwY'
componentes_ics.tgz
'https://www.drive.google.com/uc?export=download&id=1Hq0I78wqqOexmidHTeDYgmRsenNgFT5Z'
componentes_pcbs.tgz
'https://www.drive.google.com/uc?export=download&id=1-bi-PBMPL-agjGzYs6xKf0SLizqVlaJy'
componentes_reles.tgz
'https://www.drive.google.com/uc?export=download&id=1QTvG5AHlLApbckvb-m6CQtBMeA6XT8tZ'
flower_photos.tgz
'https://www.drive.google.com/uc?export=download&id=1_Dmxylc7ca0z81vtKDA1jsSPMIEnYi09'
https://drive.google.com/file/d/1_Dmxylc7ca0z81vtKDA1jsSPMIEnYi09/view?usp=sharing
componentesm.tgz
'https://www.drive.google.com/uc?export=download&id=1lYn88OxczniPwWAHbfo0h7ivBK48i9OX'
flower_photosm.tgz
'https://www.drive.google.com/uc?export=download&id=1VdUsfyMLbNxWWH4PZ1YLEmhdIhAL6LM_'
flower_photosz.tgz
https://drive.google.com/file/d/1VdUsfyMLbNxWWH4PZ1YLEmhdIhAL6LM_/view?usp=share_link
https://drive.google.com/file/d/1Dpwi4SunDdLXezF8T8N4oII5KAb3ajPI/view?usp=share_link
"""

image_path = tf.keras.utils.get_file(
      'componentesm.tgz',
      'https://www.drive.google.com/uc?export=download&id=1_Dmxylc7ca0z81vtKDA1jsSPMIEnYi09',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'componentes')

# Load input data and split into training and testing
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

# Customize TensorFlow model
model = image_classifier.create(train_data)

# Evaluate customized model
loss, accuracy = model.evaluate(test_data)

# Plot examples

def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)

  predict_label = predicts[i][0][0]
  color = get_label_color(predict_label,
                          test_data.index_to_label[label.numpy()])
  ax.xaxis.label.set_color(color)
  plt.xlabel('Predicted: %s' % predict_label)
plt.savefig("resultados")

# Export to TensorFlow Lite model
model.export(export_dir='.')

