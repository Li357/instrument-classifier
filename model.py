from os import path, makedirs
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN

from config import dropout_rate, slice_size, no_classes, cv_split, model_output, model_output_path, batch_size, no_epochs
from dataset import get_dataset

def create_model():
  cnn = input_data(name='Input', shape=[None, slice_size, slice_size, 1])

  cnn = conv_2d(cnn, 64, 2, activation='elu', weights_init='xavier')
  cnn = max_pool_2d(cnn, 2)

  cnn = conv_2d(cnn, 128, 2, activation='elu', weights_init='xavier')
  cnn = max_pool_2d(cnn, 2)

  cnn = conv_2d(cnn, 256, 2, activation='elu', weights_init='xavier')
  cnn = max_pool_2d(cnn, 2)

  cnn = conv_2d(cnn, 512, 2, activation='elu', weights_init='xavier')
  cnn = max_pool_2d(cnn, 2)

  cnn = fully_connected(cnn, 1024, activation='elu')
  cnn = dropout(cnn, dropout_rate)

  cnn = fully_connected(cnn, no_classes, activation='softmax')
  cnn = regression(cnn, n_classes=no_classes)

  return DNN(cnn)

def train_model(model):
  train_X, train_y = get_dataset('train')

  print('\nTraining model')
  model.fit(train_X, train_y, show_metric=True, validation_set=cv_split,
            shuffle=True,
            batch_size=batch_size,
            n_epoch=no_epochs)

  if not path.exists(model_output_path):
    makedirs(model_output_path)

  print('Saving model weights')
  model.save(model_output)

def test_model(model):
  print('Loading model weights')
  model.load(model_output)

  print('Testing model')
  test_X, test_y = get_dataset('test')
  accuracy = model.evaluate(test_X, test_y)[0]
  print('\nTest accuracy: {.2f}'.format(accuracy * 100))
