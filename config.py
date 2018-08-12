# Preprocessing config

dataset_path = '../../Downloads/IRMAS-TrainingData'

spectrograms_path = '../irmas_data'
slices_path = 'slices'
slice_size = 128

# Model config

pickled_dataset_path = 'pickled'

batch_size = 128
no_epochs = 20
dropout_rate = 0.5
dataset_split = 0.9
cv_split = 0.2

classes = {
  'cel': 1,  # cello
  'cla': 2,  # clarinet
  'flu': 3,  # flute
  'gac': 4,  # acoustic guitar
  'gel': 5,  # electric guitar
  'org': 6,  # organ
  'pia': 7,  # piano
  'sax': 8,  # saxophone
  'tru': 9,  # trumpet
  'vio': 10, # violin
  'voi': 11  # voice
}
no_classes = len(classes)

model_output_path = 'weights'
model_output = '{}/instrument_classifier.tflearn'.format(model_output_path)
