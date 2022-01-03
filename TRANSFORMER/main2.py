import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)

for pt_examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    pass
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    pass
    print(en.decode('utf-8'))

tokenizers = tf.saved_model.load(model_name)
encoded = tokenizers.en.tokenize(en_examples)


for row in encoded.to_list():
  print(row)


round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
  print(line.decode('utf-8'))

tokens = tokenizers.en.lookup(encoded)
print(tokens)

for i in zip(encoded, tokenizers.en.lookup(encoded)):
    print(i)


def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)


def get_data():
    # Download
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

    # Split data into train & validation
    train_examples, val_examples = examples['train'], examples['validation']

    return train_examples, val_examples


def analyze_vocab():

    train_examples, val_examples = get_data()

    # print the data samples
    for pt_examples, en_examples in train_examples.batch(5).take(1):
        # Portuguese
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))

        print("*****************************************************")

        # English
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    train_en = train_examples.map(lambda pt, en: en)
    train_pt = train_examples.map(lambda pt, en: pt)

    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )

    bert_tokenizer_params = dict(lower_case=True)
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    pt_vocab = bert_vocab.bert_vocab_from_dataset(train_pt.batch(1000).prefetch(2), **bert_vocab_args)
    en_vocab = bert_vocab.bert_vocab_from_dataset(train_en.batch(1000).prefetch(2), **bert_vocab_args)

    tokenizers = tf.saved_model.load(model_name)
    encoded = tokenizers.en.tokenize(en_examples)

    for row in encoded.to_list():
        print(row)

    txt_tokens = tf.gather(en_vocab, encoded)
    print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))

    words = tokenizers.en.detokenize(encoded)
    print(tf.strings.reduce_join(words, separator=' ', axis=-1))