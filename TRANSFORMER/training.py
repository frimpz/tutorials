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

from TRANSFORMER.Optimizer import CustomSchedule, loss_function, accuracy_function
from TRANSFORMER.models import Transformer
from TRANSFORMER.utils import make_batches, get_tokenizer_size
import argparse

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--buffer', type=int, default=20000, help='Buffer size.')
parser.add_argument('--batch', type=int, default=64, help='Batch size.')
parser.add_argument('--num_layers', type=int, default=128, help='Number od layers.')
parser.add_argument('--d_model', type=int, default=128, help='Embedding dimension.')
parser.add_argument('--dff', type=int, default=128, help='Dimension of feed forward.')
parser.add_argument('--num_heads', type=int, default=8, help='attention heads.')
parser.add_argument('--dropout_rate', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=20, help='Number of Epochs.')

args = parser.parse_args()

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']


train_batches = make_batches(train_examples, BUFFER_SIZE=args.buffer, BATCH_SIZE=args.batch)
val_batches = make_batches(val_examples, BUFFER_SIZE=args.buffer, BATCH_SIZE=args.batch)

learning_rate = CustomSchedule(d_model=args.d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


transformer = Transformer(
    num_layers=args.num_layers,
    d_model=args.d_model,
    num_heads=args.num_heads,
    dff=args.dff,
    input_vocab_size=get_tokenizer_size()[0],
    target_vocab_size=get_tokenizer_size()[1],
    pe_input=1000,
    pe_target=1000,
    rate=args.dropout_rate)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(args.epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')