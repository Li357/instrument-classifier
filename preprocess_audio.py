from subprocess import Popen, PIPE, STDOUT
from os import path, listdir, makedirs
from math import ceil
import glob
import re
import sys
import getopt
from PIL import Image
from utils import print_progressbar

from config import dataset_path, spectrograms_path, spectrogram_size, slices_path, slice_size

def generate_spectrograms():
  current_path = path.dirname(path.realpath(__file__))
  filenames = glob.glob('{}/**/*.wav'.format(dataset_path))

  for index, filename in enumerate(filenames):
    categories = re.compile('(?:\[((?:[a-z]{3}(?:_)?)+)\])').findall(filename)
    newname = '{}/{}_{}.png'.format(spectrograms_path, '_'.join(categories), index)
    cmd = 'sox {} -n spectrogram -Y {} -m -r -o {}'.format(filename, spectrogram_size, newname)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=current_path)
    output, errors = p.communicate()
    if errors:
      print(errors)
    print_progressbar(index / len(filenames))

def slice_spectrograms():
  all_spectrograms = listdir(spectrograms_path)

  if not path.exists(slices_path):
    makedirs(slices_path)

  for index, filename in enumerate(all_spectrograms):
    if filename.endswith('.png'):
      slice_spectrogram(filename)
      print_progressbar(index / len(all_spectrograms))

def slice_spectrogram(filename):
  img = Image.open('{}/{}'.format(spectrograms_path, filename))
  width, height = img.size
  no_samples = width // slice_size

  for i in range(no_samples):
    start = i * slice_size
    cropped = img.crop((start, 1, start + slice_size, slice_size + 1))
    cropped.save('{}/{}_{}.png'.format(slices_path, filename[:-4], i + 1))

def main(argv):
  help_msg = '-g: Generate spectrograms\n-s: Slice spectrograms'
  try:
    opts, args = getopt.getopt(argv, 'h:g:s')
  except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print(help_msg)
      sys.exit()
    elif opt == '-g':
      generate_spectrograms()
      print('\nDone generating spectrograms')
      sys.exit()
    elif opt == '-s':
      slice_spectrograms()
      print('\nDone slicing spectrograms')
      sys.exit()
  print(help_msg)

if __name__ == '__main__':
  main(sys.argv[1:])
