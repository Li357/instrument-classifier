from math import ceil

def print_progressbar(progress, msg=''):
  print('\r{0}: [{1:50s}] {2:.1f}%'.format(msg, '#' * int(ceil(progress * 50)), progress * 100),
                                          end='',
                                          flush=True)
