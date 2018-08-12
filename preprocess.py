from subprocess import Popen, PIPE, STDOUT
from os import path, listdir, makedirs
from math import ceil
import glob
import re
from PIL import Image
from utils import print_progressbar

from config import dataset_path, spectrograms_path, slices_path, slice_size

def generate_spectrograms():
  current_path = path.dirname(path.realpath(__file__))
  filenames = glob.glob('{}/**/*.wav'.format(dataset_path))

  for index, filename in enumerate(filenames):
    categories = re.compile('(?:\[((?:[a-z]{3}(?:_)?)+)\])').findall(filename)
    newname = '{}/{}_{}.png'.format(spectrograms_path, '_'.join(categories), index)
    cmd = 'sox {} -n spectrogram -Y {} -m -r -o {}'.format(filename, slice_size, newname)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
              close_fds=True,
              cwd=current_path)
    output, errors = p.communicate()
    if errors:
      print(errors)
    print_progressbar(index / len(filenames), 'Generating')

def slice_spectrograms():
  all_spectrograms = listdir(spectrograms_path)

  if not path.exists(slices_path):
    makedirs(slices_path)

  for index, filename in enumerate(all_spectrograms):
    if filename.endswith('.png'):
      slice_spectrogram(filename)
      print_progressbar(index / len(all_spectrograms), 'Slicing')

def slice_spectrogram(filename):
  img = Image.open('{}/{}'.format(spectrograms_path, filename))
  width, height = img.size
  no_samples = width // slice_size

  for i in range(no_samples):
    start = i * slice_size
    cropped = img.crop((start, 1, start + slice_size, slice_size + 1))
    cropped.save('{}/{}_{}.png'.format(slices_path, filename[:-4], i + 1))
