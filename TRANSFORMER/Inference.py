import argparse

import tensorflow as tf

from TRANSFORMER.models import Transformer
from TRANSFORMER.utils import get_tokenizer, get_tokenizer_size, Translator, print_translation

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