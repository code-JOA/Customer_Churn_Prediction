import tensorflow
import keras



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import data
df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values


# Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# label_encoder_X_1 = LabelEncoder()
# X[:, 1] = LabelEncoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencdoer = ColumnTransformer([("one_hot_encoder",OneHotEncoder(), [1] , remainder = "passthrough")])
# X = onehotencoder.fit_transform(X)
# X = X[:, 1:]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#onehotencoder = OneHotEncoder(categorical_features = [1])

#X = onehotencoder.fit_transform(X).toarray()

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Initialising the ANN
ann = tf.keras.models.Sequence()


# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6 , activation='relu'))
#adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6 , activation='relu'))
#adding the output layer
ann.add(tf.keras.Dense(units=6 , activation='sigmoid'))
#compiling the ANN "adam is best to update the weights"
# binary crossentropy since its a binanry target
ann.compile(optimizer = "adam" , loss="binary_crossentropy" , metrics = ['accuracy'])

# training the ANN on the training set
ann.fit(X_train , y_train , batch_size = 32 , epochs = 100)


# making a prediction using a 2d array and i set the threshold to 0.5
ann.predict(sc.transform[[1,0,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)


# predicting test results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred),1) , y_test.reshape(len(y_test),1)),1))


# Making the confusion Matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print(cm)
accuracy_score(y_test , y_pred)
