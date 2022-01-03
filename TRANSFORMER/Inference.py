import argparse

import tensorflow as tf

from TRANSFORMER.models import Transformer
from TRANSFORMER.utils import get_tokenizer, get_tokenizer_size

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

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=20):
    # input sentence is portuguese, hence adding the start and end token
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # as the target is english, the first token to the transformer should be the
    # english start token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    text = self.tokenizers.en.detokenize(output)[0]  # shape: ()

    tokens = self.tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights

tokenizers = get_tokenizer()

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

translator = Translator(tokenizers, transformer)

sentence = "este é um problema que temos que resolver."
ground_truth = "this is a problem we have to solve ."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)