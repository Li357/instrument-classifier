import numpy as np
import pandas as pd
import pickle
from os import listdir, makedirs, path
from random import shuffle
from PIL import Image

from utils import print_progressbar
from config import slice_size, slices_path, dataset_split, classes, no_classes, pickled_dataset_path

slices = listdir(slices_path)
shuffle(slices)

split_index = int(len(slices) * dataset_split)

def get_dataset(mode):
  split = slices[:split_index] if mode == 'train' else slices[split_index:]
  input_path, output_path = list(map(
    lambda template: '{}/{}{}.p'.format(pickled_dataset_path, mode, template), ['_X', '_y']))

  if (
    not path.exists(pickled_dataset_path) or
    all(not filename.endswith('.p') for filename in listdir(pickled_dataset_path))
  ):
    print('Datasets do not exist')
    try:
      # Dataset directory may not exist if not saved correctly
      makedirs(pickled_dataset_path)
    except FileExistsError:
      pass

    X, y = map_imgs_to_classes(split, mode)
    print('\nSaving dataset')
    save_datasets((X, input_path), (y, output_path))
    return X, y
  else:
    print('Datasets exist\nLoading {} dataset'.format(mode))
    return load_datasets(input_path, output_path)

def get_img_data(filename):
  img = Image.open(filename)
  data = np.asarray(img, dtype=np.uint8).reshape(slice_size, slice_size, 1) / 255
  return data

def map_imgs_to_classes(filenames, dataset_name):
  X = []
  y = []
  for index, filename in enumerate(filenames):
    X.append(get_img_data('{}/{}'.format(slices_path, filename)))
    y.append(classes[filename[:3]])
    print_progressbar(index / len(filenames), 'Building {} dataset'.format(dataset_name))
  return np.array(X), pd.get_dummies(y).values

def save_datasets(*datasets):
  pass

  # Bug in writing to files (https://bugs.python.org/issue24658) that prevents pickle from
  # serialize objects > 4 GiB on Mac OS X
  # for dataset, path in datasets:
  #   pickle.dump(dataset, open(path, 'wb'))

def load_datasets(*datasets):
  deserialized = []
  for path in datasets:
    deserialized.append(pickle.load(open(path, 'rb')))
  return deserialized
