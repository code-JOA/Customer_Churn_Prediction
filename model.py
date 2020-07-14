# Importing keras libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from sklearn.model_selection import cross_val_score , GridSearchCV


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
df_x = dataset.iloc[:, 3:13]
df_y = dataset.iloc[:, 13]

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
#X = onehotencoder.fit_transform(X)
#X = X[:, 1:]


# labelencoder_X_2 = LabelEncoder()

# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# #onehotencoder = OneHotEncoder(categorical_features = [1])

# #X = onehotencoder.fit_transform(X).toarray()

# columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

# X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

# X = X[:, 1:]




def clean_df(dataset):
    L_encoder = LabelEncoder()
    dataset["Gender"] = L_encoder.fit_transform(dataset["Gender"])
    dataset = dataset.get_dummies(data=dataset , columns=["Geography"] , drop_first=True)
    dataset = dataset.sort_index(axis=1)
    return dataset


df_x = clean_df(df_x)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
joblib.dump(X_train.columns, "columns.pkl")


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(x_test)
joblib.dump(scaler, "std_scaler.pkl")
print(X_test)
print(X_train.shape[1])


def generate_report():
    # predicting values
    y_pred = classifier.predict(x_test)
    print("\nPredicted values: " + str(y_pred) + "\n")
    y_pred = (y_pred > 0.5)
    from sklearn.metrics import plot_confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy: "+ str(accuracy) + "\n")


# Initialize the ANN
def build_model():
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6 , kernel_initializer = "uniform" , activation="relu" , input_dim=11))
    # Add the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform" , activation="relu"))
    # Add the output layer . I used sigmoid since the output variable is binary.
    classifier.add(Dense(units=1 , kernel_initializer="uniform" , activation="sigmoid"))
    # Compiling the ANN
    classifier.compile(optimizer="adam" , loss="binary_crossentropy" , metrics = ["accuracy"])
    return classifier
classifier = biuld_model()
classifier.fit(X_train , y_train , batch_size = 10  ,epochs = 100)
generate_report()


# HyperParameter Optimization
def build_model():
    classifier = Sequential()
    classifier.add(Dense(units = 6 , kernel_initializer="uniform", activation = "relu" , input_dim = X_train.shape[1]))
    classifier.add(Dense(units= 6 , kernel_initializer="uniform" , activation = "relu"))
    classifier.add(Dense(units= 1 , kernel_initializer="uniform" , activation = "sigmoid"))
    classifier.add(Dense(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_model)
parameters = {"batch_size" : [5,15,30] , "epochs":[30,100] , "optimizer":["adam","rmsprop"]}
grid_search = GridSearchCV(esrtimator=classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv=10)
grid_search = grid_search.fit(X_train , y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters , best_accuracy)
generate_rep()


# save the model
joblib.dump(classifier , "prediction_classifier.pkl")
