# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Designing and implementing a neural network regression model aims to accurately predict a continuous target variable based on a set of input features from the provided dataset. The neural network learns complex relationships within the data through interconnected layers of neurons. The model architecture includes an input layer for the features, several hidden layers with non-linear activation functions like ReLU to capture complex patterns, and an output layer with a linear activation function to produce the continuous target prediction.

## Neural Network Model

![363492719-d5fcd281-7218-43e5-b823-e796503665e7](https://github.com/user-attachments/assets/0f23b836-911a-49c4-9f00-c3d79e8a0096)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Varsha G
### Register Number:212222230166
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Ai_Brain = Sequential([
    Dense(units = 9, activation = 'relu',input_shape = [8]),
    Dense(units = 9, activation = 'relu'),
    Dense(units = 9, activation = 'relu'),
    Dense(units = 1)
])

Ai_Brain.summary()

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('varsha').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df.head()

dataset1 = pd.DataFrame(rows[1:],columns=rows[0])
dataset1 = dataset1.astype({'input':'int'})
dataset1 = dataset1.astype({'output':'int'})

X = dataset1[['input']].values
y = dataset1[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state=0)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

Ai_Brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

X_train1 = Scaler.transform(X_train)

Ai_Brain.compile(optimizer = 'rmsprop', loss='mse')

Ai_Brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(Ai_Brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

Ai_Brain.evaluate(X_test1,y_test)

X_n1 = [[3]]

X_n1_1 = Scaler.transform(X_n1)

Ai_Brain.predict(X_n1_1)


```
## Dataset Information

![image](https://github.com/user-attachments/assets/1b6644a9-0ae4-4929-9453-bdb719352c03)


## OUTPUT



### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/af504b64-5200-4705-af5a-4cf00d842eb0)

### Test Data Root Mean Squared Error

![Screenshot 2024-09-01 201231](https://github.com/user-attachments/assets/103091ab-f378-4c4d-84a2-fa04f499262a)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/355871dd-bde9-4b53-9056-1679f7317c7b)


## RESULT

Thus, Neural network for Regression model is successfully Implemented
