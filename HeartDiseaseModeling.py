
# Author:       Ethan Pattison
# FSU Course:   SENG 609
# Professor:    Dr Abusharkh
# Assingment:   Assignment 10: Final Project
# Date:         10/12/2022


# import packages and functions

import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.compose import make_column_transformer
import joblib

# Link the Database to the Python file so the program has the information needed
# Save the Data as a dataframe

df = pd.read_csv('heart_2020.csv', sep=',')

df


# Create Function to Change the Columns to Numeric Values.

def tran_HeartDisease(a):
    if a == "No":
        return 0
    if a == "Yes":
        return 1

    # Use the function and add the columns to the dataframe


df['HeartDiseaseNum'] = df['HeartDisease'].apply(tran_HeartDisease)
df['SmokingNum'] = df['Smoking'].apply(tran_HeartDisease)
df['AlcoholDrinkingNum'] = df['AlcoholDrinking'].apply(tran_HeartDisease)
df['StrokeNum'] = df['Stroke'].apply(tran_HeartDisease)
df['DiffWalkingNum'] = df['DiffWalking'].apply(tran_HeartDisease)
df['PhysicalActivityNum'] = df['PhysicalActivity'].apply(tran_HeartDisease)
df['AsthmaNum'] = df['Asthma'].apply(tran_HeartDisease)
df['KidneyDiseaseNum'] = df['KidneyDisease'].apply(tran_HeartDisease)
df['SkinCancerNum'] = df['SkinCancer'].apply(tran_HeartDisease)

df


# Define a function to normalize

def normalize(Var):
    x = np.array(df[Var]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    X_scaled = scaler.transform(x)
    df[Var] = X_scaled.reshape(1, -1)[0]


# Use the Function to normalize Physical and Mental Heath Ratings

normalize("MentalHealth")
normalize("PhysicalHealth")

# Print to See if the the values are changed
df.head(9)


# Get the Columns and out from the file and save as a dataframe

XY = df[['Sex', 'HeartDiseaseNum', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
         'DiffWalkingNum', 'PhysicalActivityNum', 'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]


# One Hot-Encode for the column Sex

columns_trans = make_column_transformer(
    (OneHotEncoder(), ["Sex"]),
    remainder='passthrough'
)
X = columns_trans.fit_transform(XY)
print(X)


# Convert the Data back to a dataframe

Data = pd.DataFrame(X,
                    columns=['Female', 'Male', 'HeartDiseaseNum', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum',
                             'PhysicalHealth', 'MentalHealth',
                             'DiffWalkingNum', 'PhysicalActivityNum', 'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum',
                             'SkinCancerNum'])
Data


# Set the columns for X and Y

x = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

y = Data[['HeartDiseaseNum']]


# Linear Regression 3 Different Testing Splits 20/25/30

# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Create a Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make prediction later
joblib.dump(model, 'HeartDisease.pkl')

# Report how well the model is performing
print("Linear Regression training results (20% Testing): ")

mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create a Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make prediction later
joblib.dump(model, 'HeartDisease.pkl')

# Report how well the model is performing
print("Linear Regression training results (25% Testing): ")

mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Create a Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make prediction later
joblib.dump(model, 'HeartDisease.pkl')

# Report how well the model is performing
print("Linear Regression training results (30% Testing): ")

mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# Decision Trees 3 Different Testing Splits 20/25/30

from sklearn import tree

# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

joblib.dump(clf, 'HeartDisease1.pkl')

# Report how well the model is performing
print("Decision Tree results (20% Testings): ")

mse_train = mean_absolute_error(y_train, clf.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, clf.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

joblib.dump(clf, 'HeartDisease1.pkl')

# Report how well the model is performing
print("Decision Tree results (25% Testings): ")

mse_train = mean_absolute_error(y_train, clf.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, clf.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

joblib.dump(clf, 'HeartDisease1.pkl')

# Report how well the model is performing
print("Decision Tree results (30% Testings): ")

mse_train = mean_absolute_error(y_train, clf.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, clf.predict(X_test))
print(f" - Testing Set Error: {mse_test}")


# KNN Modeling

x = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

y = Data[['HeartDiseaseNum']]

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=0)

# Import the Scaler before using KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Import and train the model
from sklearn.neighbors import KNeighborsClassifier

classifer = KNeighborsClassifier(n_neighbors=5)
classifer.fit(X_train, y_train.values.ravel())

joblib.dump(classifer, 'HeartDiseaseKNN.pkl')


# Run the classifier to get the predictions for comparison (Testing)
y_pred = classifer.predict(X_test)

# Display a Confusion Matrix to find the accuracy
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Run the classifier to get the predictions for comparison (Training)
y_prediction = classifer.predict(X_train)

# Display a Confusion Matrix to find the accuracy
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_train, y_prediction))
print(confusion_matrix(y_train, y_prediction))


# install Tensorflow and Keras to your machine

import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# Neural Network Variation 1

# Disable pandas warning
pd.options.mode.chained_assignment = None

# Create X and Y arrays
x = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

y = Data[['HeartDiseaseNum']]

# Scale the data to (1-0) for the Neural Network
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both training inputs and outputs
x[x.columns] = x_scaler.fit_transform(x[x.columns])
y[y.columns] = y_scaler.fit_transform(y[y.columns])

# Tensorflow does not support df so convert into numpy arrays
x = x.to_numpy()
y = y.to_numpy()

# Split the data into training and testing for the modeling (25% - Testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create the model from Keras / Use Sequentail API
model = Sequential()

# Input layer / feed 13 inputs / 50 nodes
model.add(Dense(50, input_dim=13, activation='relu'))

# hidden layer with 100 nodes
model.add(Dense(100, activation='relu'))

# Add Layer with 200 nodes
model.add(Dense(200, activation='relu'))

# hidden layer with 100 nodes
model.add(Dense(100, activation='relu'))

# Output Layer for Predicted house value
model.add(Dense(1, activation='linear'))

# Have keras construct the neural network inside of Tensorflow
model.compile(loss='mean_squared_error', optimizer='SGD')

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=8,
    shuffle=True,
    verbose=1
)

# Save the scalers to files so we can use it to preprocess new data later
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Save the trained model to a file so we can use it to make predictions later
model.save("HeartDiseasePredictionNN.h10")

# Report how well model is performing
print("Model training results: ")

# Report Mean absolute error on training
predictions_train = model.predict(x_train, verbose=0)

# Convert the data back to the origanal form for the prediction for training
mse_train = mean_absolute_error(
    y_scaler.inverse_transform(predictions_train),
    y_scaler.inverse_transform(y_train)
)

print(f" - Training Set Error: {mse_train}")

# Report Mean absolute error on training
predictions_test = model.predict(x_test, verbose=0)

# Convert the data back to the origanal form for the prediction for testing
mse_test = mean_absolute_error(
    y_scaler.inverse_transform(predictions_test),
    y_scaler.inverse_transform(y_test)
)

print(f" - Testing Set Error: {mse_test}")



# Neural Network Variation 2

# Disable pandas warning
pd.options.mode.chained_assignment = None

# Create X and Y arrays
x = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

y = Data[['HeartDiseaseNum']]

# Scale the data to (1-0) for the Neural Network
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both training inputs and outputs
x[x.columns] = x_scaler.fit_transform(x[x.columns])
y[y.columns] = y_scaler.fit_transform(y[y.columns])

# Tensorflow does not support df so convert into numpy arrays
x = x.to_numpy()
y = y.to_numpy()

# Split the data into training and testing for the modeling (25% - Testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Create the model from Keras / Use Sequentail API
model = Sequential()

# Input layer / feed 13 inputs / 100 nodes
model.add(Dense(100, input_dim=13, activation='relu'))

# hidden layer with 100 nodes
model.add(Dense(100, activation='relu'))

# hidden layer with 200 nodes
model.add(Dense(200, activation='relu'))

# hidden layer with 400 nodes
model.add(Dense(400, activation='relu'))

# Add Layer with 600 nodes
model.add(Dense(600, activation='relu'))

# Add Layer with 100 nodes
model.add(Dense(100, activation='relu'))

# Output Layer for Predicted house value
model.add(Dense(1, activation='linear'))

# Have keras construct the neural network inside of Tensorflow
model.compile(loss='mean_squared_error', optimizer='SGD')

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=500,
    shuffle=True,
    verbose=1
)

# Save the scalers to files so we can use it to preprocess new data later
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Save the trained model to a file so we can use it to make predictions later
model.save("HeartDiseasePredictionNN.h10")

# Report how well model is performing
print("Model training results: ")

# Report Mean absolute error on training
predictions_train = model.predict(x_train, verbose=0)

# Convert the data back to the origanal form for the prediction for training
mse_train = mean_absolute_error(
    y_scaler.inverse_transform(predictions_train),
    y_scaler.inverse_transform(y_train)
)

print(f" - Training Set Error: {mse_train}")

# Report Mean absolute error on training
predictions_test = model.predict(x_test, verbose=0)

# Convert the data back to the origanal form for the prediction for testing
mse_test = mean_absolute_error(
    y_scaler.inverse_transform(predictions_test),
    y_scaler.inverse_transform(y_test)
)

print(f" - Testing Set Error: {mse_test}")


# Example of program using KNN

def main_Function():
    # Application to repeat the model for people that come into the clinic

    # Introduction to the program
    print("\nHello and welcome to the Healthy Heart Predictor!")

    print("\nThis program will ask the user 12 simple qustions and predict if the user is at risk for Heart disease: ")

    # Add Spacing
    print("\n")

    # Get all the information from the 12 Questions
    Attribute1 = int(input("\nAre you a male: (1 for Yes or 0 for No) "))
    Attribute3 = int(input("\nHave you smoked 100 cigarettes or more throughout your life: (1 for Yes or 0 for No) "))
    Attribute4 = int(input(
        "\nDo you average 1 alcoholic drink per day if female or 2 alcoholic drinks per day if male: (1 for Yes or 0 for No) "))
    Attribute5 = int(input("\nHave you ever been told you had a stroke: (1 for Yes or 0 for No) "))
    Attribute6 = int(
        input("\nHow many physical illness's and injury's have occured in the last Month: (Range is 0-30) "))
    Attribute7 = int(
        input("\nHow many days during the past 30 days was your mental health not good?: (Range is 0-30) "))
    Attribute8 = int(input("\nDo you have serious difficulty walking or climbing stairs?: (1 for Yes or 0 for No) "))
    Attribute9 = int(
        input("\nPhysical activity or exercise during the past 30 days other than their regular job:(Range is 0-30) "))
    Attribute10 = int(input("\nOn average, how many hours of sleep do you get in a 24-hour period?: "))
    Attribute11 = int(input("\nHave you ever been told you has asthma?: (1 for Yes or 0 for No) "))
    Attribute12 = int(input(
        "\nNot including kidney stones or bladder infections, were you ever told you had kidney disease? : (1 for Yes or 0 for No) "))
    Attribute13 = int(input("\nHave you ever been told you had skin cancer?: (1 for Yes or 0 for No) "))

    # Create a if statement to generate if Attribute 2 is female or not
    if Attribute1 == 1:
        Attribute2 = 0
    else:
        Attribute2 = 1

    # Add Spacing
    print("\n")

    # Load in the model
    model = joblib.load('HeartDiseaseKNN.pkl')

    # Save the information from the person in a 2d array
    Person_1 = [[Attribute1, Attribute2, Attribute3, Attribute4, Attribute5,
                 Attribute6, Attribute7, Attribute8, Attribute9, Attribute10,
                 Attribute11, Attribute12, Attribute13]]

    # Make a prediction for the person
    Heart_Prediction = model.predict(Person_1)

    # We are prediction the first row in the array (0)
    predicted_value = Heart_Prediction[0]

    print("\nEstimation for Heart disease ranges from 0 to 1.")
    print("The Lower the estimation, the lower the chance you have of having or obtaining a heart disease.")

    # print the results
    print(f"\nYour Estimation for heart disease: {predicted_value}")

    if predicted_value >= .7:
        print(
            "\nYou are at a very high risk for Heart Disease, adjust lifestyle and see a medical professional as soon as possible.")

    elif predicted_value >= .5 and predicted_value < .7:
        print("\nYou are at risk for Heart Disease, please consider adjusting lifestyle")

    else:
        print("\nContinue to live the way you are, no adjustment need to be made.")
        print("Your body is at a low risk for Heart Disease.")

    restart = input("\nWould you like to analyze another person?: ").lower()
    if restart == "yes":
        print("\n")
        main_Function()
    else:
        print("\nHave a great day!")


# Run the main function

main_Function()





