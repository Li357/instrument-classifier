import numpy as np
import pandas as pd
from os import listdir
from random import shuffle
from PIL import Image

from utils import print_progressbar
from config import slice_size, slices_path, dataset_split, classes, no_classes

slices = listdir(slices_path)
shuffle(slices)

split_index = int(len(slices) * dataset_split)

def load_dataset(mode):
  split = slices[:split_index] if mode == 'train' else slices[split_index:]
  return map_imgs_to_classes(split, mode)

def get_img_data(filename):
  img = Image.open(filename)
  data = np.asarray(img, dtype=np.uint8).reshape(slice_size, slice_size, 1) / 255
  return data

def map_imgs_to_classes(filenames, dataset_name):
  X = np.array([get_img_data('{}/{}'.format(slices_path, filename)) for filename in filenames])
  y = pd.get_dummies([classes[filename[:3]] for filename in filenames]).values
  # X = X.reshape([len(filenames), slice_size, slice_size, 1])
  return X, y
