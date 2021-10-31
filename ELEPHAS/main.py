from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
sc = SparkContext(conf=conf)
from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
# from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


x_train, y_train = load_iris(return_X_y=True)
print(type(x_train))
print(type(y_train))
# exit()

from elephas.utils.rdd_utils import to_simple_rdd
rdd = to_simple_rdd(sc, x_train, y_train)

print(rdd.collect())


from elephas.spark_model import SparkModel
from tensorflow.keras.callbacks import History

history = History()
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
pip = spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1, callbacks=[history])

print(pip)
print(history.history)