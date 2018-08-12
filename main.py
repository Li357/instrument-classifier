import sys
import getopt
from model import create_model, train_model, test_model
from preprocess import generate_spectrograms, slice_spectrograms

def main(argv):
  help_msg = '''
    -g: Generate spectrograms
    -s: Slice spectrograms
    -c: Create and train model
    -t: Test model
  '''

  try:
    opts, args = getopt.getopt(argv, 'hgsct')
  except getopt.GetoptError as err:
    print(err)
    print(help_msg)
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print(help_msg)
    elif opt == '-g':
      generate_spectrograms()
      print('\nDone generating spectrograms')
    elif opt == '-s':
      slice_spectrograms()
      print('\nDone slicing spectrograms')
    elif opt in ('-c', '-t'):
      model = create_model()
      print('Done creating model')
      if opt == '-c':
        train_model(model)
        print('Done training model')
      else:
        test_model(model)
        print('Done testing model')
  print(help_msg)

if __name__ == '__main__':
  main(sys.argv[1:])
