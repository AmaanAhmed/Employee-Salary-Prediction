#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Approach for Employee Salary Prediciton

# #### Steps Involved:
#     1. Importing libraries and Dataset.
#     2. Data preprocessing.
#     3. Explanatory Data analysis.
#     4. Feature Engineering.
#     5. Predictive Modelling and Validation.
#     6. Optimization.

# ### Part 1 - Importing Libraries and Dataset

# In[664]:


# In this section, We will import all the libraries required and the dataset we will be using for this project
# We will update this section along the way and add libraries we require as per our needs

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


# In[573]:


# Now let's import the dataset

data = pd.read_csv(r'C:\Users\Amaan Ahmed\Music\Project\Employee_Payroll.csv')

# This project is available on my GitHub.
# If you cloned my repository, you will need to change this path to the one where the dataset is located


# In[574]:


# Let's check our dataset using the head() method. This step is optional and in no way does it affect our goal.

data.head(5)


# ### Part 2 - Data Preprocessing

# In[575]:


# Let's calculate the number of null values present in the columns of our dataset

data.isnull().sum()


# In[576]:


# Let's find out the number of rows or instances in our dataset

len(data)


# In[577]:


# Let's determine the percentage of null values present in the columns of our dataset

null_percentage = (data.isnull().sum() / len(data)) * 100
print(null_percentage)


# In[578]:


# Now, observbe that none of our columns have more than 80% of null values.
# Hence we can choose to fill these null values instead of dropping the entire rows.
# Therefore, we can use the forward fill method to fill those null values.

data.fillna(method='ffill', inplace=True)


# In[579]:


# Now let's use the info() method to check if the previous step worked or not

data.info()


# In[580]:


# As we can see, it worked. None of the columns contains a null value.
# But, if you observe, all the column names has a space between them
# Also the original hire date column, which contains an actual date, is of 'object' data type.
# Let's correct these two issues.

# Let's start with the column names

data.columns


# In[581]:


# Let's change the column names using the rename() method

data.rename(columns = lambda x: x.replace(" ","_"), inplace = True)


# In[582]:


# Let's check and confirm the changes

data.columns


# In[583]:


# Now let's change the datatype of the original hire date

# import datetime

data['Original_Hire_Date'] = pd.to_datetime(data['Original_Hire_Date'])


# In[584]:


# Let's check and confirm the changes

data.info()


# In[585]:


# Let's also drop duplicates as our last step in this section

data.drop_duplicates(inplace=True)


# ### Part 3 - Exploratory Data Analysis

# In[586]:


# Lets take a look at our dataset

data.head(5)


# In[587]:


# Now, notice that we have a feature (column) in our dataset called "Employee_identifier".
# This feature contains unique identification numbers of the employees.
# We can use this feature to determine the number of employees we have in our dataset

data.Employee_Identifier.nunique()


# In[588]:


# We know that this dataset contains information about 28375 employees.
# Now let's once again take a look at the number of instances in our dataset

len(data)


# In[589]:


# Notice that our dataset contains 234298 instances.
# Let's filter our dataset and select only those instances that contain duplicates of employees identification number.
# Performing this operation we will be only selecting the employees who have repeated instances.

data[data.Employee_Identifier.duplicated() == True]


# In[590]:


# It is clear from the above result that the data is only for 28375 employees,
# but they all have multiple instances that add upto 205923.
# Now upon close inspection, we can also notice that the data is spread across multiple fiscal periods of the years.
# Our goal is to determine the salary of an employee and we need to figure out the base pay per year.
# This information is not available in our dataset, instead it is broken down into multiple quarters of the year.
# So in this case, let's introduce a new feature (column) called "year".
# We can extract the year from the original_hire_date feature and add it to our new feature.

data['year'] = data['Original_Hire_Date'].dt.year


# In[591]:


# Now let's check if it worked before we move forward.

data.head(5)


# In[592]:


# Now let's target the "base_pay" feature. First, We will group our dataset by the "year" feature.
# Then, we will calculate the average of the "base_pay" feature and plot it on a bar diagram for better understanding.
# Also, we will display only the largest 10 and smallest 10 results only.

# import matplotlib.pyplot as plt

data.groupby('year')['Base_Pay'].mean().nlargest(10).plot(kind = 'bar')


# In[593]:


data.groupby('year')['Base_Pay'].mean().nsmallest(10).plot(kind = 'bar')


# In[594]:


# From the above two cells, we can see the largest 10 and smallest 10 results,
# which are nothing but the highest and lowest average values of the base pay.
# Now let's sort our "year" column.

data['year'].sort_values(ascending=True)


# In[595]:


# It is clear from the above result that our dataset contains 60 years of data.
# Now, let's create a line plot showing the relationship between years and their corresponding base pay.
# In our line plot, let the x-axis represent the years, sorted in ascending order, and the y-axis represent the base pay.
# Let the plot have a red line indicating the trend.
# Finally, we will use a dark grid style for the plot background.

# import seaborn as sns

plt.figure(figsize=(12,8))
sns.set_style('darkgrid')
with sns.axes_style('dark'):
    sns.lineplot(x=data['year'].sort_values(ascending=True), y= data['Base_Pay'], color = 'red')
plt.show()


# In[596]:


# We can notice which decade had the highest and lowest average base pay from the above result.
# Now let's find out which job titles are paid the highest and which ones are paid the lowest.
# First, let's see how many unique jobs are present in our dataset

data.Job_Title.nunique()


# In[597]:


# There are 2382 number of unique jobs.
# Now, using the same logic we used earlier, let's calculate the average base pay according to the job titles.
# We will calculate for both highest and lowest.

data.groupby('Job_Title')['Base_Pay'].mean().nlargest(20)


# In[598]:


data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)


# In[599]:


# The results for the average highest base pay is fine as can be seen above.
# But there is a problem in the average lowest base pay.
# The average of some job titles is 0.
# This is a problem. Maybe the dataset do not contain the "base_pay" information for these job titles.
# Let's fix this issue.

# First let's see how many instances we have with 0 base pay.

data[data.Base_Pay == 0]


# In[600]:


# There are 2541 instances with no information about the base_pay
# Let's remove these rows from our dataframe.
# There are some different ways to do so but the one I find easy to understand is:

data = data[data.Base_Pay != 0]

# In the above line, we are extracting the instances whose base_pay feature does not have 0 
# and adding them back into the same dataframe.


# In[601]:


# Let's check the number of instances now.

data.shape


# In[602]:


# Let's repeat the previous steps.

data.Job_Title.nunique()


# In[603]:


data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)


# In[604]:


# We have fixed this issue successfully.
# Now, # lets see how the yearly payment is divided in quarters.
# First, we will find out how many times each value in the "fiscal_quarter" feature occurs.

data.Fiscal_Quarter.value_counts()


# In[605]:


# In our dataset, the base pay of the employees is spread across fiscal quarters.
# So, let's determine the average base pay for each of these fical quarters.

average_per_quarter = data.groupby('Fiscal_Quarter')['Base_Pay'].mean()
average_per_quarter


# In[606]:


# Let us generate a pie chart for better understanding.

label = ['Quarter_1','Quarter_2','Quarter_3','Quarter_4' ] 

plt.figure(figsize=(10,6))
plt.pie(average_per_quarter, labels=label, startangle=90)


# In[607]:


# It is easily comprehensible that the base pay is almost equal in all the fiscal quarters.
# Now, for the next step let's repeat the same steps and try to figure out similar statistics for each month.

# To do so, we will first introduce a column that shows the month of hire.
# This step is similar to introducing the "year" column.

data['Month_of_hire'] = data['Original_Hire_Date'].dt.month


# In[608]:


# We know that the "month_of_hire" column contains the date in numeric format.
# Therefore, we can assume that the new column we just created holds numeric values.
# We can check by calling the head() method.

data.head(5)


# In[609]:


# Let's alter the month_of_hire column to show the name of the month instead of the chronogical number of that month.

data['Month_of_hire'] = pd.to_datetime(data['Month_of_hire'], format='%m').dt.strftime('%B')


# In[610]:


# Let's check to see if it worked.

data.head(5)


# In[611]:


# Now, let's introduce one more column.
# In this new column we will calculate and store the monthly base pay of the employee.

data['Monthly_Pay'] = data['Base_Pay'] /12


# In[612]:


# Now, let's verify the changes again.

data.head(5)


# In[613]:


# Now, we can find out in which month most of the people are hired.
# By determining this, we can find out more about the peak hiring season.

# First, let's count the number of times the month_of_hire values are repeated.
# This will give us an idea about how many people are hired in those months.

data.Month_of_hire.value_counts()


# In[614]:


# From above result, it is clear that the month of june has the highest number of hires
# and the month of january has the lowest number of hires.
# It is safe to assume that the third quarter of the year is the season where the hiring is at peak.

# Now, let's find out the average base pay for each month.
# Let's also plot a bar chart for better understanding.

data.groupby('Month_of_hire')['Monthly_Pay'].mean().sort_values(ascending = False).plot(kind = 'bar', edgecolor = 'black' )


# In[615]:


# From above visualization, it is clear that the months of July, august and september has the highest average base pay.
# This is the same quarter where the hiring is at peak.
# Can we assume that the new hires in the third quarter are being offered larger base pay packages?
# Let's explore a little more. Let's take a look at other features in our dataset.

data.head(10)


# In[616]:


# We have the 'bureau' feature.
# There are many offices under each department.
# Let's determine the number of offices under each bureau.
# We will also determine the average base pay with respect to the offices.
# Doing so we will ultimately figure out the average base pay of each department.

# We will achieve this in three steps.
# First, we will group the data by Bureau column.
# Then, we will count the number of offices for each value in bureau column.
# Lastly, we will calculate the average of base pay for each department.
# we will perform these three steps together.

data.groupby('Bureau').aggregate({'Office':'count', 'Base_Pay':'mean'}).nlargest(15,columns=['Office'])


# In[617]:


# The above result shows the number of offices for each department and their respective average base pay.
# Similarly, we can keep on going and explore the data in many other ways.
# We can use other features as the center of attraction as we did with "base_pay" column and perform similar operations.
# In this way, we are open to several possibilities of exploring the dataset.
# But, our goal is to build a model that predicts the "base_pay". So we kept our operations around that column.

# Let's generate a histogram to check for the distribution of the base_pay column.

plt.figure(figsize=(12,8))
sns.distplot(data['Base_Pay'], color='red')


# In[618]:


# The histogram provides a visual representation of the distribution of base pay values.
# We can see how frequently different ranges of base pay occurs in the dataset.

# Now let's also check for outliers before moving to next section.

sns.boxplot(data['Base_Pay'])


# In[619]:


# It is clearly visible that our dataset does contain some outliers.
# But in my opinion, they seem to be negligeable.
# So, let's conclude our exploratory data analysis section here and move forward without worrying about outliers.

# Notice that we made a few of changes to the dataset.
# We explored and introduced two more columns.


# ### Part 4 - Feature Engineering

# In[620]:


# Let's start by creating an extra copy of our dataset.
# We will do so in case we make changes that can't be reversed.

data2 = data


# In[621]:


# Verify the previous step

data2.head()


# In[622]:


# Now, let's start by dropping the columns that are not relevant to our goal.
# Let's drop irrelevant columns and keep only those that are relevant.
# We can do so by using the drop method. But I think there is a more eaier way to do it.
# We can simply select and copy the columns we need from the original dataset.

data2 = data[['Fiscal_Year', 'Fiscal_Quarter', 'Office','Job_Code', 'Base_Pay', 'Position_ID', 'year','Month_of_hire', 'Monthly_Pay']]
data2


# In[623]:


# Now, let's check for collinearity between the columns in our new dataset

plt.figure(figsize=(12,10))
sns.heatmap(data2.corr(), linecolor='black', linewidths=2.0, annot=True, cmap='Blues_r')
plt.xticks(fontsize = 12)


# In[624]:


# Notice that the monthly_pay column and the base_pay column are both collinear.
# This is because we derived the monthly_pay column from the base_pay column.
# Let's remove the base_pay column from our dataset and repeat the above step.
# Also base_pay column is our target value so we will separate it and store it in a variable.

y = data2.pop('Base_Pay')
x = data2 

# We removed the base_pay column and stored it in variable 'y'.
# We stored the rest of the dataset in variable 'x'.

# Now, we repeat the previous step.

plt.figure(figsize=(12,10))
sns.heatmap(data2.corr(), linecolor='black', linewidths=2.0, annot=True, cmap='Blues_r')
plt.xticks(fontsize = 12)


# In[625]:


# Now, there is no such heavy collinearity in our dataset.
# Also, we separated our target value from the dataset so from here on we will use the x dataframe.
# Let's check our dataframe.

x.head(5)


# In[626]:


x.info()


# In[627]:


# Here, notice that the month_of_hire is a categorical column which includes 12 categories (Months of the year).
# We can break this column down into 12 separate columns and use each of them as a boolean type instead of object string.
# First we will create a dataframe and extract each of the value from "month_of_hire" column.
# We will insert these values into new dataframe as columns.

x_dummies = pd.get_dummies(data['Month_of_hire'], prefix = 'Month',prefix_sep='_',drop_first=True)
x_dummies


# In[628]:


# Now let's join this dataframe to our dataframe and drop the "month_of_hire" column.

x = x.join(x_dummies)
x.drop(columns=['Month_of_hire'], inplace=True)


# In[629]:


x.head(5)


# In[630]:


x.info()


# In[631]:


# Notice that we created 12 new columns and each of these columns has an int datatype with 8 bits.
# These new columns contain two values 0 and 1.
# 0 meaning False and 1 meaning True.
# we eliminated the object string and made it a little easier to perform operations.

# Now let's standardize the dataset.

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# The above line creates a new instance of StandardScaler Class.
# The StandardScaler is used to standardize features by removing the mean and scaling to unit variance.


# In[632]:


# Let's standardize the features stored in x

x[x.columns] = scaler.fit_transform(x[x.columns])

# Let me explain the above line. Here's what happens:
# First, the 'x.columns' returns the column names of x.
# Then, 'x[x.columns]' selects all the columns in x.
# After that, the 'scaler.fit_transform(x[x.columns])' part scales all the selected columns.
# The resulting scaled values are then assigned back to the respective columns in x using the same 'x[x.columns]'.


# In[633]:


# Let's reshape the y dataframe.

y = y.values.reshape(-1,1)

# The above line will convert the single column in the y dataframe into a 2D array.


# In[634]:


# Let's scale the values in y dataframe as well.
# This way we will ensure that the target values are also scaled to unit values.

y = scaler.fit_transform(y)


# In[635]:


# By performing the previous steps, we eliminated any chance of biases among the data in our columns.
# All the columns noiw contain unit values.
# Let's move to the next step which is to build a predictive model.


# ### Part 5 - Predictive Modeling and Validation

# In[636]:


# First we will need to split our dataset into training data and testing data.

# from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.75, random_state=42,shuffle=True)

# Here we extracted 75% of data randomly and created a training datset with it.
# The remaining 25% data will automatically be assigned as testing data.


# In[637]:


# Let's check the 4 dataframes we just created

x_train.shape


# In[638]:


x_test.shape


# In[639]:


y_train.shape


# In[640]:


y_test.shape


# In[641]:


# Now that our training data and testing data is ready, Let's proceed to building a model.
# First let's perform Linear Regression.

# from sklearn.linear_model import LinearRegression

# Creating an model instance of the LinearRegression class.

Linear_model = LinearRegression()

# Fitting the model with the training data.

Linear_model.fit(x_train, y_train)

# Let's use the predict method to make the model predict our target values.

y_pred_linear = Linear_model.predict(x_test)


# In[642]:


# Now, Let's calculate few statistics and print them.

# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
rmse = mean_squared_error(y_test, y_pred_linear, squared=False)
mae = mean_absolute_error(y_test, y_pred_linear)

print("The R-Squared Score is: " + str(r2))
print("The Mean Squared Error is: " + str(mse))
print("The Root Mean Squared Error is: " + str(rmse))
print("The Mean Absolute Error is: " + str(mae))


# In[643]:


# Let's Validate by comparing the R-Squared score of the model's predicted data to the actual data.

r2_train = r2_score(y_train, Linear_model.predict(x_train))
r2_train


# In[644]:


# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_linear)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# In[645]:


# Let's repeat the above steps but with a different model.
# Let's experiment with Support Vector Regressor

# from sklearn.svm import SVR

model_svr = SVR()

model_svr.fit(x_train,y_train)


# In[646]:


y_pred_svr = model_svr.predict(x_test)

r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print("The R-Squared Score is: " + str(r2_svr))
print("The Mean Squared Error is: " + str(mse_svr))
print("The Root Mean Squared Error is: " + str(rmse_svr))
print("The Mean Absolute Error is: " + str(mae_svr))

# Validating the same way
print(" ")
print("Validating SVR: ")
r2_train_svr = r2_score(y_train, model_svr.predict(x_train))
r2_train_svr


# In[647]:


# Here, the R2 score of training dataset is slightly more than that of testing dataset which means that model is performing accurately.
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_svr)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# In[648]:


# Let's repeat the above steps again but with a different model.
# Let's experiment with Decision Tree Regressor

# from sklearn.tree import DecisionTreeRegressor

model_DTR = DecisionTreeRegressor()

model_DTR.fit(x_train,y_train)


# In[649]:


y_pred_DTR = model_DTR.predict(x_test)

r2_DTR = r2_score(y_test, y_pred_DTR)
mse_DTR = mean_squared_error(y_test, y_pred_DTR)
rmse_DTR = mean_squared_error(y_test, y_pred_DTR, squared=False)
mae_DTR = mean_absolute_error(y_test, y_pred_DTR)

print("The R-Squared Score is: " + str(r2_DTR))
print("The Mean Squared Error is: " + str(mse_DTR))
print("The Root Mean Squared Error is: " + str(rmse_DTR))
print("The Mean Absolute Error is: " + str(mae_DTR))

# Validating the same way
print(" ")
print("Validating Decision Tree: ")
r2_train_DTR = r2_score(y_train, model_DTR.predict(x_train))
r2_train_DTR


# In[650]:


# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_DTR)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# In[651]:


# Let's repeat the above steps but with a different model.
# Let's experiment with Random Forest Regression.

# from sklearn.ensemble import RandomForestRegressor

model_RF = RandomForestRegressor()

model_RF.fit(x_train,y_train)


# In[653]:


y_pred_RF = model_RF.predict(x_test)

r2_RF = r2_score(y_test, y_pred_RF)
mse_RF = mean_squared_error(y_test, y_pred_RF)
rmse_RF = mean_squared_error(y_test, y_pred_RF, squared=False)
mae_RF = mean_absolute_error(y_test, y_pred_RF)

print("The R-Squared Score is: " + str(r2_RF))
print("The Mean Squared Error is: " + str(mse_RF))
print("The Root Mean Squared Error is: " + str(rmse_RF))
print("The Mean Absolute Error is: " + str(mae_RF))

# Validating the same way
print(" ")
print("Validating Random Forest: ")
r2_train_RF = r2_score(y_train, model_RF.predict(x_train))
r2_train_RF


# In[654]:


# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_RF)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# ### Part 6 - Optimization

# In[657]:


# Notice that out of all the models we have trained, Random Forest has the lowest MSE with corresponding R2 score.
# Therefore, Let's Optimize Random Forest Regression model.
# Let's use GridSearchCV for Model Tuning.

# from sklearn.model_selection import GridSearchCV
# from sklearn import ensemble

model_t = GridSearchCV(ensemble.RandomForestRegressor(), {'n_estimators': [10,15, 30], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 4],},cv=3)
model_t.fit(x_train,y_train)


# In[658]:


# Let's breakdown the above two lines to understadn what is being done here.

# 'ensemble.RandomForestRegressor()`: This part creates an instance of a RandomForestRegressor model
# This instance is an ensemble learning method for used regression tasks.

# `{'n_estimators': [10,15, 30], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 4],}`
# This part is a dictionary specifying the hyperparameter grid to search.
# It includes different values for 'n_estimators' (number of trees in the forest), 'max_depth' (maximum depth of the trees), and 'min_samples_leaf' (minimum samples required to be at a leaf node).

# `cv=3` This part specifies the number of folds for cross-validation during the grid search.
# In this case, we've set it to 3, meaning the data will be split into 3 subsets for cross-validation.

# `model_t.fit(x_train, y_train)`: This line fits the GridSearchCV model to the training data:


# In[659]:


# Now that we have performed the GridSearchCV, Let's print out some information we need.

print(model_t.best_estimator_)
print(model_t.best_params_)
print(model_t.best_score_)


# In[660]:


# We have found the best parameters and now we can use these to optimize our model.
# Let's repeat the Random Forest Regression steps with optimized parameters.

final_model = RandomForestRegressor(max_depth=20, n_estimators=30, min_samples_leaf=1)

final_model.fit(x_train,y_train)


# In[661]:


predicted = final_model.predict(x_test)

r2_Final = r2_score(y_test, predicted)
mse_Final = mean_squared_error(y_test, predicted)
rmse_Final = mean_squared_error(y_test, predicted, squared=False)
mae_Final = mean_absolute_error(y_test, predicted)

print("The R-Squared Score is: " + str(r2_Final))
print("The Mean Squared Error is: " + str(mse_Final))
print("The Root Mean Squared Error is: " + str(rmse_Final))
print("The Mean Absolute Error is: " + str(mae_Final))

# Validating the same way
print(" ")
print("Validating the optimized Random Forest: ")
r2_train_Final = r2_score(y_train, final_model.predict(x_train))
r2_train_Final


# In[662]:


# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,predicted)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()


# In[666]:


# Notice that it took a really long time for GridSearchCV to train the model.
# Let's save these models so we won't have to repeat the whole process.

# import joblib as jb

# Saving the optimized Random Forest Model:

jb.dump(final_model, filename= r'C:\Users\Amaan Ahmed\Music\Project\RF_model.pkl')

# Saving the Grid Search Model:

jb.dump(model_t, filename=r'C:\Users\Amaan Ahmed\Music\Project\grid_estimator.pkl')


# # END OF PROJECT

# In[ ]:




