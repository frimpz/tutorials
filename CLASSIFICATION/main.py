## Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow import feature_column

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



import pathlib

dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                        extract=True, cache_dir='.')
dataframe = pd.read_csv(csv_file)
dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

# dataframe = dataframe[['Type', 'Age', 'Fee', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health']]
print(dataframe.head(3))

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensors((dict(dataframe), labels))
  return ds
  if shuffle:
     ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# data = tfds.as_numpy(train_ds)
# for i in data:
#   print(i)


# for i in train_ds.as_numpy_iterator():
#   print(i)


feature_columns = []

# numeric cols
for header in ['PhotoAmt', 'Fee', 'Age']:
  feature_columns.append(feature_column.numeric_column(header))


# bucketized cols
age = feature_column.numeric_column('Age')
age_buckets = feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5])
feature_columns.append(age_buckets)


# indicator_columns
indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                          'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, dataframe[col_name].unique())
  indicator_column = feature_column.indicator_column(categorical_column)
  feature_columns.append(indicator_column)


# embedding columns
breed1 = feature_column.categorical_column_with_vocabulary_list(
      'Breed1', dataframe.Breed1.unique())
breed1_embedding = feature_column.embedding_column(breed1, dimension=8)
feature_columns.append(breed1_embedding)


animal_type = feature_column.categorical_column_with_vocabulary_list(
      'Type', ['Cat', 'Dog'])

# crossed columns
age_type_feature = feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(age_type_feature))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


class BardModel(Model):
  def __init__(self):
    super(BardModel, self).__init__()
    self.bardmodel = tf.keras.Sequential([
      feature_layer,
      layers.Dense(128, activation=tf.nn.relu),
      # layers.Dropout(0.2),
      layers.Dense(128, activation=tf.nn.relu),
      # layers.Dropout(0.2),
      # layers.Dense(64, activation=tf.nn.relu),
      layers.Dropout(0.1),
      layers.Dense(1, activation='softmax'),
    ])


  def call(self, x):
    final = self.bardmodel(x)
    return final


norm = 1.0
pos_weight = 1.0
class ErrorFunction(Loss):

  def __init__(self, pos_weight, norm ):
    super(ErrorFunction, self).__init__()
    self.pos_weight = pos_weight
    self.norm = norm

  def call(self, y_true, y_pred):
    print(y_true)
    print(y_pred)
    # y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return self.norm * tf.reduce_mean(
      tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred, labels=y_true, pos_weight=self.pos_weight))


model = BardModel()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
cost = ErrorFunction(pos_weight, norm)
model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # loss=cost,
              metrics=[metrics.Accuracy()]#, metrics.Precision(), metrics.Recall(), metrics.AUC()]
              )


history = model.fit(train_ds, epochs=10, shuffle=True, validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy is  {} & Loss is {}".format(accuracy, loss))

# example_batch, lbs = next(iter(train_ds))
#
# for i in zip(model.bardmodel(example_batch).numpy(), lbs):
#     print(i)

print(train_ds)
exit()
example_batch, lbs = train_ds
print(lbs)
exit()
# for i in zip(model.bardmodel(example_batch).numpy(), lbs):
#     print(i)

print(model.bardmodel.summary())


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
