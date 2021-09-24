import tensorflow as tf
import numpy as np


def text_to_dataset(file_path, seq_len, buffer_size, batch_size):

  # Read, then decode for py2 compat.
  text = open(file_path, 'rb').read().decode(encoding='utf-8')
  # length of text is the number of characters in it
  print('Length of text: {} characters'.format(len(text)))

  # Take a look at the first 250 characters in text
  print(" looking at the first 250 characters of the text...", text[:250])

  # The unique characters in the file
  vocab = sorted(set(text))
  print ('{} unique characters...'.format(len(vocab)))
  vocab_size = len(vocab)

  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text_as_int = np.array([char2idx[c] for c in text])

  print('{')
  for char,_ in zip(char2idx, range(20)):
      print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
  print('  ...\n}')

  examples_per_epoch = len(text)//(seq_len + 1)

  # Create training examples / targets
  char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

  sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)

  for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

  def split_input_target(chunk):
      input_text = chunk[:-1]
      target_text = chunk[1:]
      return input_text, target_text

  dataset = sequences.map(split_input_target)

  dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

  return dataset, vocab_size

if __name__ == "__main__":
  file_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

  file_path='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/shakespeare_short.txt'

  BATCH_SIZE = 64
  BUFFER_SIZE = 10000
  seq_len=10
  dataset, vocab_size=text_to_dataset(file_path=file_path, seq_len=seq_len, buffer_size=BUFFER_SIZE, batch_size=64)
  print('dataset', dataset)




