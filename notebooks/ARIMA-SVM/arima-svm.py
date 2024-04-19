import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import pickle
from sklearn import svm

warnings.filterwarnings('ignore')


df = pd.read_csv('C:/Users/91738/Desktop/CogNight/input/sleepdata_2.csv')
df['Sleep Quality'] = df['Sleep Quality'].str.rstrip('%').astype('float')
df = df[df['Sleep Quality']>40]
data = df['Sleep Quality'].copy().reset_index().drop('index',axis=1)
df[['Date','Time']] = df['Start'].str.split(expand=True)
data = df[['Date','Sleep Quality']].copy().reset_index().drop('index',axis=1)

y = (data['Sleep Quality']).to_numpy()
y

### ARIMA Model for Predicting Sleep Score (Efficiency)

arima = pickle.load(open('C:/Users/91738/Desktop/CogNight/models/arima_fit.pkl','rb'))
arima.summary()

for i in range(3):
    arima_fit = arima.extend(y)

pred = arima_fit.forecast(1)
pred


### SVM ###

dfml = pd.read_csv('C:/Users/91738/Desktop/CogNight/input/Sleep_Efficiency.csv')

dfml = dfml.drop(['Bedtime', 'Wakeup time', 'ID','REM sleep percentage','Deep sleep percentage','Light sleep percentage','Awakenings'] ,axis=1)
dfml['Smoking status'] = dfml['Smoking status'].map({'Yes':1 ,'No':0})
dfml['Gender'] = dfml['Gender'].map({'Male':1 ,'Female':0})
cols = [i for i in dfml.columns if i not in ["Sleep efficiency","Sleep duration"]]
dfml = dfml.dropna()

for col in cols:
  dfml[col] = dfml[col].astype(int)
  

dfml = dfml.reset_index().drop('index',axis=1)
dfml['Caffeine consumption'] = (dfml['Caffeine consumption']/25).astype(int)

dfml = dfml[['Gender', 'Age', 'Smoking status', 'Alcohol consumption','Caffeine consumption', 'Exercise frequency', 'Sleep efficiency','Sleep duration']]
dfml

X = dfml.iloc[:,:-1]
Y = dfml.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.svm import SVR

svm = SVR(kernel='linear')
svm.fit(X,Y)

y_pred = svm.predict(X_test)
err = mean_squared_error(y_test, y_pred)
print(err)

plt.plot(y_pred)
plt.plot(list(y_test))
plt.title("Error in prediction (hrs)")
plt.legend()
plt.show()


from sklearn.svm import SVR

svm = SVR(kernel='rbf')
svm.fit(X,Y)

y_pred = svm.predict(X_test)
err = mean_squared_error(y_test, y_pred)
print(err)

plt.plot(y_pred)
plt.plot(list(y_test))
plt.title("Error in prediction (hrs)")
plt.legend(['y_pred','y_test'])
plt.show()

### COMBINING THE RESULTS 

print("Insert your sleep quality for last 5 days (out of 100%).")
sq_in = []
for i in range(5):
    sq_in.append(int(input()))

sq_in = np.array(sq_in)

for i in range(3):
    arima_fit = arima_fit.extend(sq_in)

sq = arima_fit.forecast(1)
print("Predicted sleep quality for today is: ",sq)

age = int(input("Age:"))
age = (69-age)/60
gender = int(input("Gender(1 for male, 0 for female):"))
caff = int(input("Caffeine(on a scale of 0 to 10):"))
alc = int(input("Alcohol(on a scale of 0 to 5):"))
smoke = int(input("Smoking status(0 for No, 1 for Yes):"))
exer = int(input("Exercise(on a scale of 0 to 5:)"))

'Gender', 'Age', 'Smoking status', 'Alcohol consumption','Caffeine consumption', 'Exercise frequency'
user_input = np.array([gender, age, smoke, alc, caff, exer, sq/100])
user_input = np.reshape(user_input,(1,-1))

y_pred_2 = svm.predict(user_input)

print("You need to sleep for: %.2f Hours" % (y_pred_2.item()))
