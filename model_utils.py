import tensorflow.keras as keras

def define_model():
	# Build DNN model with keras
	model = keras.Sequential()
	model.add(keras.Input(shape=(99,)))
	model.add(keras.layers.Dense(units=64, activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(units=16, activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dense(units=5, activation='softmax'))
	# model.summary()
	return model 

def load_model(model):
	return keras.models.load_model("action.h5")
