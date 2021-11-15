'''
    Code using pandas Dataframe
'''

from tensorflow.keras.datasets import mnist
from pyspark import RDD, SparkContext
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import tensorflow as tf
from Classes import DataGenerator

from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
sc = SparkContext(conf=conf)


def to_simple_rdd(sc: SparkContext, features: np.array, labels: np.array):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def to_spark_df(x, y):
    df = pd.DataFrame(x)
    df['label'] = y.tolist()
    df.insert(0, 'person_id', df.index)
    df = df.head(1000)
    return df


def add_label(x, y):
    df = pd.DataFrame(x)
    df['label'] = y.tolist()
    # df.insert(0, 'person_id', df.index)
    # df = df.set_index('person_id')
    return df

spark = SparkSession.builder.getOrCreate()
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

train = to_spark_df(x_train, y_train)
test = to_spark_df(x_test, y_test)



spark = SparkSession.builder.getOrCreate()
train_rdd = spark.createDataFrame(train)
# print(train_rdd.show())

test_rdd = spark.createDataFrame(test)
# print(test_rdd.show())

# partition -> Can get this from spark dataframe
partition = {
    'train': train['person_id'].tolist(),
}
# labels -> Can get this dictionary from the task gold standard.
labels = train.set_index('person_id').to_dict()['label']



# Parameters
params = {'dim': 784,
          'batch_size': 50,
          'n_classes': 10,
          'shuffle': True,
          'data': train_rdd}


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)


model = tf.keras.Sequential([
       tf.keras.layers.Dense(128, input_dim=784, activation='relu'),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'])


model.fit(x=training_generator,
                    # validation_data=validation_generator,
                    # use_multiprocessing=True,
                    workers=6, epochs=5)
