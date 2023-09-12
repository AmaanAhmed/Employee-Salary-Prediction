import pandas as pd
import numpy as np 

import warnings
warnings.filterwarnings('ignore')

import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import joblib as jb

data = pd.read_csv(r'C:\Users\Amaan Ahmed\Music\Project\Employee_Payroll.csv')

# Part 2 - Data Preprocessing

data.isnull().sum()

null_percentage = (data.isnull().sum() / len(data)) * 100
print(null_percentage)

data.fillna(method='ffill', inplace=True)

data.rename(columns = lambda x: x.replace(" ","_"), inplace = True)

data['Original_Hire_Date'] = pd.to_datetime(data['Original_Hire_Date'])

data.drop_duplicates(inplace=True)


# Part 3 - Exploratory Data Analysis

data.Employee_Identifier.nunique()

data[data.Employee_Identifier.duplicated() == True]

data['year'] = data['Original_Hire_Date'].dt.year

data.groupby('year')['Base_Pay'].mean().nlargest(10).plot(kind = 'bar')

data.groupby('year')['Base_Pay'].mean().nsmallest(10).plot(kind = 'bar')

data['year'].sort_values(ascending=True)

data.Job_Title.nunique()

data.groupby('Job_Title')['Base_Pay'].mean().nlargest(20)

data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)

data[data.Base_Pay == 0]

data = data[data.Base_Pay != 0]

data.Job_Title.nunique()

data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)

data.Fiscal_Quarter.value_counts()

average_per_quarter = data.groupby('Fiscal_Quarter')['Base_Pay'].mean()
average_per_quarter

data['Month_of_hire'] = data['Original_Hire_Date'].dt.month

data['Month_of_hire'] = pd.to_datetime(data['Month_of_hire'], format='%m').dt.strftime('%B')

data['Monthly_Pay'] = data['Base_Pay'] /12

data.Month_of_hire.value_counts()

data.groupby('Month_of_hire')['Monthly_Pay'].mean().sort_values(ascending = False).plot(kind = 'bar', edgecolor = 'black' )

data.groupby('Bureau').aggregate({'Office':'count', 'Base_Pay':'mean'}).nlargest(15,columns=['Office'])

# Part 4 - Feature Engineering


data2 = data


data2.head()

data2 = data[['Fiscal_Year', 'Fiscal_Quarter', 'Office','Job_Code', 'Base_Pay', 'Position_ID', 'year','Month_of_hire', 'Monthly_Pay']]
data2

plt.figure(figsize=(12,10))
sns.heatmap(data2.corr(), linecolor='black', linewidths=2.0, annot=True, cmap='Blues_r')
plt.xticks(fontsize = 12)

y = data2.pop('Base_Pay')
x = data2 

plt.figure(figsize=(12,10))
sns.heatmap(data2.corr(), linecolor='black', linewidths=2.0, annot=True, cmap='Blues_r')
plt.xticks(fontsize = 12)

x_dummies = pd.get_dummies(data['Month_of_hire'], prefix = 'Month',prefix_sep='_',drop_first=True)
x_dummies

x = x.join(x_dummies)
x.drop(columns=['Month_of_hire'], inplace=True)

scaler = StandardScaler()

x[x.columns] = scaler.fit_transform(x[x.columns])

y = y.values.reshape(-1,1)

y = scaler.fit_transform(y)

# Part 5 - Predictive Modeling and Validation

x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.75, random_state=42,shuffle=True)

Linear_model = LinearRegression()

Linear_model.fit(x_train, y_train)

y_pred_linear = Linear_model.predict(x_test)

r2 = r2_score(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
rmse = mean_squared_error(y_test, y_pred_linear, squared=False)
mae = mean_absolute_error(y_test, y_pred_linear)

print("The R-Squared Score is: " + str(r2))
print("The Mean Squared Error is: " + str(mse))
print("The Root Mean Squared Error is: " + str(rmse))
print("The Mean Absolute Error is: " + str(mae))

r2_train = r2_score(y_train, Linear_model.predict(x_train))
r2_train

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_linear)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

model_svr = SVR()

model_svr.fit(x_train,y_train)

y_pred_svr = model_svr.predict(x_test)

r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print("The R-Squared Score is: " + str(r2_svr))
print("The Mean Squared Error is: " + str(mse_svr))
print("The Root Mean Squared Error is: " + str(rmse_svr))
print("The Mean Absolute Error is: " + str(mae_svr))

print(" ")
print("Validating SVR: ")
r2_train_svr = r2_score(y_train, model_svr.predict(x_train))
r2_train_svr

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_svr)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

model_DTR = DecisionTreeRegressor()

model_DTR.fit(x_train,y_train)

y_pred_DTR = model_DTR.predict(x_test)

r2_DTR = r2_score(y_test, y_pred_DTR)
mse_DTR = mean_squared_error(y_test, y_pred_DTR)
rmse_DTR = mean_squared_error(y_test, y_pred_DTR, squared=False)
mae_DTR = mean_absolute_error(y_test, y_pred_DTR)

print("The R-Squared Score is: " + str(r2_DTR))
print("The Mean Squared Error is: " + str(mse_DTR))
print("The Root Mean Squared Error is: " + str(rmse_DTR))
print("The Mean Absolute Error is: " + str(mae_DTR))

print(" ")
print("Validating Decision Tree: ")
r2_train_DTR = r2_score(y_train, model_DTR.predict(x_train))
r2_train_DTR

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_DTR)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

model_RF = RandomForestRegressor()

model_RF.fit(x_train,y_train)

y_pred_RF = model_RF.predict(x_test)

r2_RF = r2_score(y_test, y_pred_RF)
mse_RF = mean_squared_error(y_test, y_pred_RF)
rmse_RF = mean_squared_error(y_test, y_pred_RF, squared=False)
mae_RF = mean_absolute_error(y_test, y_pred_RF)

print("The R-Squared Score is: " + str(r2_RF))
print("The Mean Squared Error is: " + str(mse_RF))
print("The Root Mean Squared Error is: " + str(rmse_RF))
print("The Mean Absolute Error is: " + str(mae_RF))

print(" ")
print("Validating Random Forest: ")
r2_train_RF = r2_score(y_train, model_RF.predict(x_train))
r2_train_RF

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_RF)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

# Part 6 - Optimization

model_t = GridSearchCV(ensemble.RandomForestRegressor(), {'n_estimators': [10,15, 30], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 4],},cv=3)
model_t.fit(x_train,y_train)

print(model_t.best_estimator_)
print(model_t.best_params_)
print(model_t.best_score_)

final_model = RandomForestRegressor(max_depth=20, n_estimators=30, min_samples_leaf=1)

final_model.fit(x_train,y_train)

predicted = final_model.predict(x_test)

r2_Final = r2_score(y_test, predicted)
mse_Final = mean_squared_error(y_test, predicted)
rmse_Final = mean_squared_error(y_test, predicted, squared=False)
mae_Final = mean_absolute_error(y_test, predicted)

print("The R-Squared Score is: " + str(r2_Final))
print("The Mean Squared Error is: " + str(mse_Final))
print("The Root Mean Squared Error is: " + str(rmse_Final))
print("The Mean Absolute Error is: " + str(mae_Final))

print(" ")
print("Validating the optimized Random Forest: ")
r2_train_Final = r2_score(y_train, final_model.predict(x_train))
r2_train_Final

plt.figure(figsize=(10,6))
plt.scatter(y_test,predicted)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

jb.dump(final_model, filename= r'C:\Users\Amaan Ahmed\Music\Project\RF_model.pkl')

jb.dump(model_t, filename=r'C:\Users\Amaan Ahmed\Music\Project\grid_estimator.pkl')

# END OF PROJECT
