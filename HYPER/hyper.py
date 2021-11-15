

'''
    Function to create a keras model.
'''
from keras.wrappers.scikit_learn import KerasClassifier
def create_model():
    model = None
    return model

model = KerasClassifier(build_fn=create_model)