from math import ceil

def print_progressbar(progress):
  print('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * int(ceil(progress * 50)), progress * 100),
    end='', flush=True)
