import numpy
import pandas
import os
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import model_from_json

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load INPUTS
INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '/')
TRAIN_IMAGES_DIR = os.path.join(INPUTS_DIR, 'dataset/iris.csv')

#Load ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
args, unparsed = parser.parse_known_args()
epochs = args.epochs
batch_size = args.batch_size

# load dataset
dataframe = pandas.read_csv(TRAIN_IMAGES_DIR, header=None, engine='python')
dataset = dataframe.values
X = dataset[1:,0:4].astype(float)
Y = dataset[1:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model  
estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
  
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

outputs_dir = os.getenv('VH_OUTPUTS_DIR', '/')
output_file = os.path.join(outputs_dir, 'my_model.h5')
print('Saving model to %s' % output_file)
model_json = model.to_json()
with open(output_file, "w") as json_file:
    json_file.write(model_json)
#model.model.save_weights(output_file)
#files = os.listdir(output_dir)
#print('The file is here %s' %files[0])
