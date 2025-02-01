# importing  libraries
import tensorflow as tf #framework
import pandas as pd # convert dataset to dataframe
import numpy as np # numercial value
from sklearn.model_selection import train_test_split #input and output splitting to train and test
from sklearn.preprocessing import StandardScaler

#create a sample dataset
data ={
    'YEAR':[2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'MILEAGE':[20,23,23,19,18,18,17,19,15],
    'HP':[100,120,140,160,180,200,220,240,260],
    'PRICE':[10000,12000,14000,16000,18000,20000,22000,24000,26000]
}
#converting the dataset to dataframe
ds = pd.DataFrame(data)


#define feature (x) and target (y)
x = ds[['YEAR', 'MILEAGE', 'HP']] #independent
y = ds['PRICE'] #dependent

#spliting the date to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#normalize the feature
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#normalize the target
y_train_max=y_train.max()
y_train=y_train/y_train_max
y_test=y_test/y_train_max

#define the model mse-meansquared error,mae-mean absolute error
model=tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation='relu',input_shape=(x_train.shape[1],)),  #neural network
    tf.keras.layers.Dense(64,activation='relu'),#rectified linear unit *
    tf.keras.layers.Dense(1)
])

#compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae']) #Mean Squared Error =measures the average of error squares #Mean Absolute Error =measure the accuracy


# prediction input
input_y=input("year")
input_m=input("mileage")
input_h=input("hp")

#train the model
model.fit(x_train,y_train,epochs=100,verbose=1)

#evaluate the model
loss,mae=model.evaluate(x_test,y_test,verbose=1)
print(f"test loss(mse):{loss:4f},Test mae:{mae:4f}") #consistency

# making a prediction
input_arr=np.array([[input_y,input_m,input_h]])
input_arr=scaler.transform(input_arr)
prediction=model.predict(input_arr)
predicted_price=prediction[0][0]*y_train_max
print(f"predicted price:$ {predicted_price:,.2f}") #consistency