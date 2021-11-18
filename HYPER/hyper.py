'''
    Function to create a keras model.
'''
import numpy
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import Precision


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# create model
def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode='uniform',
                 act_1='relu', act_2='relu', act_3='relu', act_4='relu',
                 act_5='relu', act_6='relu', act_7='relu', metrics=None,
                 dropout_rate_1 = 0.0, dropout_rate_2 = 0.0, dropout_rate_3 = 0.0, dropout_rate_4 = 0.0,
                 dropout_rate_5 = 0.0, dropout_rate_6 = 0.0, dropout_rate_7 = 0.0):

    model = Sequential()
    model.add(Dense(4582, input_dim=8, kernel_initializer=init_mode, activation=act_1))
    model.add(Dropout(dropout_rate_1))
    model.add(Dense(916, kernel_initializer=init_mode, activation=act_2))
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(183, kernel_initializer=init_mode, activation=act_3))
    model.add(Dropout(dropout_rate_3))
    model.add(Dense(36, kernel_initializer=init_mode, activation=act_4))
    model.add(Dropout(dropout_rate_4))
    model.add(Dense(18, kernel_initializer=init_mode, activation=act_5))
    model.add(Dropout(dropout_rate_5))
    model.add(Dense(9, kernel_initializer=init_mode, activation=act_6))
    model.add(Dropout(dropout_rate_6))
    model.add(Dense(3, kernel_initializer=init_mode, activation=act_7))
    model.add(Dropout(dropout_rate_7))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))

    # Compile model
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  # optimizer=optimizer,
                  optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
                  metrics=metrics)
    return model


################ --- Grid Search --- ####################
seed = 7
numpy.random.seed(seed)

# load dataset
dataset = numpy.loadtxt("data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


params = {
    'verbose' : 1,
}
model = KerasClassifier(build_fn=create_model, **params)

# define the grid search parameters
# batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform']
act_1 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_2 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_3 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_4 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_5 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_6 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
act_7 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
'''dropout_rate_1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_4 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_5 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_6 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_rate_7 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
'''

param_grid = dict(
              # batch_size=batch_size,
              epochs=epochs,
              #optimizer=optimizer,
              learn_rate=learn_rate,
              # momentum=momentum,
              init_mode=init_mode,
              act_1=act_1,
              act_2=act_2,
              act_3=act_3,
              act_4=act_4,
              act_5=act_5,
              act_6=act_6,
              act_7=act_7,
              #dropout_rate_1=dropout_rate_1, dropout_rate_2=dropout_rate_2,
              #dropout_rate_3=dropout_rate_3, dropout_rate_4=dropout_rate_4,
              #dropout_rate_5=dropout_rate_5, dropout_rate_6=dropout_rate_6,
              #dropout_rate_7=dropout_rate_7,
                  )
grid_params = {
    'scoring': ['accuracy', 'precision'],
    'refit': 'precision',
    'n_jobs': -1,
    'cv': 3,
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, **grid_params)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_precision']
stds = grid_result.cv_results_['std_test_precision']
params = grid_result.cv_results_['params']
zipped = zip(means, stds, params)
zipped = sorted(zipped, key = lambda x: x[0], reverse=True)
for mean, stdev, param in zipped:
    print("%f (%f) with: %r" % (mean, stdev, param))


############ How to Tune the Training Optimization Algorithm ##########
