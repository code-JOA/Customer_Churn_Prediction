# Importing keras libraries
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6 , kernel_initializer = 'uniform' , activation='relu' , input_dim=11))

# Add the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform' , activation='relu'))

# Add the output layer
classifier.add(Dense(units=1 , kernel_initializer='uniform' , activation='sigmoid'))
