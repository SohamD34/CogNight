import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings('ignore')


data = pd.read_csv('C:/Users/91738/Desktop/CogNight/input/data (2).csv')

X,y = data.iloc[:,:-1],data.iloc[:,-1]

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
y = y.to_numpy()

data


### SPLITTING THE DATASET

train, test = train_test_split(y, train_size=29975/30000, shuffle=False)

### SARIMAX

# # Finding the best set of parameters p,d,q

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA

# d,q = 0,0         # we shall vary p

# for p in range(1,6):
#     model = ARIMA(train[:-1], order=(p, d, q))  # Replace p, d, q with appropriate values
#     model_fit = model.fit()

#     forecast = model_fit.forecast(steps=1)
#     prediction = forecast[0]
#     actual = train[-1]

#     error = abs(actual-prediction)

#     plt.scatter(p,error)

# plt.xlabel('p')
# plt.ylabel('error')
# plt.show()

# # Finding the best set of parameters p,d,q

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA

# p,q = 1,0         # we shall vary d

# for d in range(1,6):
#     model = ARIMA(train[:-1], order=(p, d, q))  # Replace p, d, q with appropriate values
#     model_fit = model.fit()

#     forecast = model_fit.forecast(steps=1)
#     prediction = forecast[0]
#     actual = train[-1]

#     error = abs(actual-prediction)

#     plt.scatter(d,error)

# plt.xlabel('d')
# plt.ylabel('error')
# plt.show()

# # Finding the best set of parameters p,d,q

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA

# p,d = 1,2         # we shall vary q

# for q in range(1,6):
#     model = ARIMA(train[:-1], order=(p, d, q))  # Replace p, d, q with appropriate values
#     model_fit = model.fit()

#     forecast = model_fit.forecast(steps=1)
#     prediction = forecast[0]
#     actual = train[-1]

#     error = abs(actual-prediction)

#     plt.scatter(q,error)

# plt.xlabel('q')
# plt.ylabel('error')
# plt.show()

best_p,best_d,best_q = 5,4,2

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


arima = ARIMA(train, order=(best_p, best_d, best_q))  # Replace p, d, q with appropriate values
arima_fit = arima.fit()

print(arima_fit.summary())


import numpy as np

actual = y[-25:]
pred = []

for i in range(25):
    try:
        forecast = arima_fit.forecast(steps=1)
        prediction = forecast[0]
        print(i, "Actual =", actual[i], "Predicted =", prediction)
        pred.append(prediction)

        train = np.append(train, actual[i])

        model = ARIMA(train, order=(best_p, best_d, best_q))
        arima_fit = model.fit()
        
    except np.linalg.LinAlgError:
        
        model = ARIMA(train, order=(best_p, best_d, best_q))
        arima_fit = model.fit(method_kwargs={'enforce_stationarity': False, 'enforce_invertibility': False})

pred = [69.59956748525835, 71.96011976823415, 69.9518183702866, 74.1454855074901, 72.72686638353218, 71.5, 70.58694590108897, 67.07988826850459, 68.14773324804752, 64.61943253298135, 66.42995870743968, 67.2064764502748, 67.62303727106982, 72.65865526904082, 72.90916826328808, 69.58479618835617, 69.83417999971644, 69.17118373540164, 66.65180214798073,  67.37771812605517,  68.95414991935064, 71.53583405099427, 72.93126566152911, 70.80764767463378, 71.61473868999423]

plt.plot(pred)
plt.plot(actual)
plt.ylim(0,100)
plt.title("Tracking the actual and predicted values")
plt.legend(["Predicted","Actual"])
plt.show()


plt.plot(pred-actual)
plt.plot(np.mean(pred-actual)*np.ones(25))
plt.title("Error graph")
plt.legend(["Error","Mean Error"])
plt.show()


import pickle
arima_fit.save('arima_model.pkl')
