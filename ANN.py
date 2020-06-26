# Importing keras libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
#X = onehotencoder.fit_transform(X)
#X = X[:, 1:]


labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#onehotencoder = OneHotEncoder(categorical_features = [1])

#X = onehotencoder.fit_transform(X).toarray()

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

X = X[:, 1:]




def clean_df(df):
    L_encoder = LabelEncoder()
    df['Gender'] = L_encoder.fit_transform(df['Gender'])
    df = df.get_dummies(data=df , columns=["Geography"] , drop_first=True)
    df = df.sort_index(axis=1)
    return df


df_new = clean_df(df_new)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(x_test)


#save model with joblib
model_scaler_file = open("standardscaler.pkl" , "wb")
joblib.dump(sc , model_scaler_file)
model_file.close()
print(x_test)
print(X_train.shape[1])




# Initialize the ANN
def biuld_ANN_model():
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6 , kernel_initializer = 'uniform' , activation='relu' , input_dim=11))
    # Add the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform' , activation='relu'))
    # Add the output layer . I used sigmoid since the output variable is binary.
    classifier.add(Dense(units=1 , kernel_initializer='uniform' , activation='sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics = ['accuracy'])
    return classifier


classifier = biuld_ANN_model()
classifier.fit(X_train , y_train , batch_size = 10  ,epochs = 100)
generate_report()
