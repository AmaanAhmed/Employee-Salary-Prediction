# Machine Learning Approach for Employee Salary Prediciton

#### Steps Involved:
    1. Importing libraries and Dataset.
    2. Data preprocessing.
    3. Explanatory Data analysis.
    4. Feature Engineering.
    5. Predictive Modelling and Validation.
    6. Optimization.

### Part 1 - Importing Libraries and Dataset


```python
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
```


```python
# Now let's import the dataset

data = pd.read_csv(r'C:\Users\Amaan Ahmed\Music\Project\Employee_Payroll.csv')

# This project is available on my GitHub.
# If you cloned my repository, you will need to change this path to the one where the dataset is located
```


```python
# Let's check our dataset using the head() method. This step is optional and in no way does it affect our goal.

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal Year</th>
      <th>Fiscal Quarter</th>
      <th>Fiscal Period</th>
      <th>First Name</th>
      <th>Last Name</th>
      <th>Middle Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office Name</th>
      <th>Job Code</th>
      <th>Job Title</th>
      <th>Base Pay</th>
      <th>Position ID</th>
      <th>Employee Identifier</th>
      <th>Original Hire Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>05/16/2005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>05/16/2005</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>05/16/2005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>05/16/2005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>05/16/2005</td>
    </tr>
  </tbody>
</table>
</div>



### Part 2 - Data Preprocessing


```python
# Let's calculate the number of null values present in the columns of our dataset

data.isnull().sum()
```




    Fiscal Year                0
    Fiscal Quarter             0
    Fiscal Period              0
    First Name                 0
    Last Name                  0
    Middle Init            93412
    Bureau                     0
    Office                  3184
    Office Name             3184
    Job Code                   0
    Job Title                  0
    Base Pay                   4
    Position ID                0
    Employee Identifier        0
    Original Hire Date         0
    dtype: int64




```python
# Let's find out the number of rows or instances in our dataset

len(data)
```




    234299




```python
# Let's determine the percentage of null values present in the columns of our dataset

null_percentage = (data.isnull().sum() / len(data)) * 100
print(null_percentage)
```

    Fiscal Year             0.000000
    Fiscal Quarter          0.000000
    Fiscal Period           0.000000
    First Name              0.000000
    Last Name               0.000000
    Middle Init            39.868715
    Bureau                  0.000000
    Office                  1.358947
    Office Name             1.358947
    Job Code                0.000000
    Job Title               0.000000
    Base Pay                0.001707
    Position ID             0.000000
    Employee Identifier     0.000000
    Original Hire Date      0.000000
    dtype: float64
    


```python
# Now, observbe that none of our columns have more than 80% of null values.
# Hence we can choose to fill these null values instead of dropping the entire rows.
# Therefore, we can use the forward fill method to fill those null values.

data.fillna(method='ffill', inplace=True)
```


```python
# Now let's use the info() method to check if the previous step worked or not

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 234299 entries, 0 to 234298
    Data columns (total 15 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   Fiscal Year          234299 non-null  int64  
     1   Fiscal Quarter       234299 non-null  int64  
     2   Fiscal Period        234299 non-null  object 
     3   First Name           234299 non-null  object 
     4   Last Name            234299 non-null  object 
     5   Middle Init          234299 non-null  object 
     6   Bureau               234299 non-null  object 
     7   Office               234299 non-null  float64
     8   Office Name          234299 non-null  object 
     9   Job Code             234299 non-null  int64  
     10  Job Title            234299 non-null  object 
     11  Base Pay             234299 non-null  float64
     12  Position ID          234299 non-null  int64  
     13  Employee Identifier  234299 non-null  object 
     14  Original Hire Date   234299 non-null  object 
    dtypes: float64(2), int64(4), object(9)
    memory usage: 26.8+ MB
    


```python
# As we can see, it worked. None of the columns contains a null value.
# But, if you observe, all the column names has a space between them
# Also the original hire date column, which contains an actual date, is of 'object' data type.
# Let's correct these two issues.

# Let's start with the column names

data.columns
```




    Index(['Fiscal Year', 'Fiscal Quarter', 'Fiscal Period', 'First Name',
           'Last Name', 'Middle Init', 'Bureau', 'Office', 'Office Name',
           'Job Code', 'Job Title', 'Base Pay', 'Position ID',
           'Employee Identifier', 'Original Hire Date'],
          dtype='object')




```python
# Let's change the column names using the rename() method

data.rename(columns = lambda x: x.replace(" ","_"), inplace = True)
```


```python
# Let's check and confirm the changes

data.columns
```




    Index(['Fiscal_Year', 'Fiscal_Quarter', 'Fiscal_Period', 'First_Name',
           'Last_Name', 'Middle_Init', 'Bureau', 'Office', 'Office_Name',
           'Job_Code', 'Job_Title', 'Base_Pay', 'Position_ID',
           'Employee_Identifier', 'Original_Hire_Date'],
          dtype='object')




```python
# Now let's change the datatype of the original hire date

# import datetime

data['Original_Hire_Date'] = pd.to_datetime(data['Original_Hire_Date'])
```


```python
# Let's check and confirm the changes

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 234299 entries, 0 to 234298
    Data columns (total 15 columns):
     #   Column               Non-Null Count   Dtype         
    ---  ------               --------------   -----         
     0   Fiscal_Year          234299 non-null  int64         
     1   Fiscal_Quarter       234299 non-null  int64         
     2   Fiscal_Period        234299 non-null  object        
     3   First_Name           234299 non-null  object        
     4   Last_Name            234299 non-null  object        
     5   Middle_Init          234299 non-null  object        
     6   Bureau               234299 non-null  object        
     7   Office               234299 non-null  float64       
     8   Office_Name          234299 non-null  object        
     9   Job_Code             234299 non-null  int64         
     10  Job_Title            234299 non-null  object        
     11  Base_Pay             234299 non-null  float64       
     12  Position_ID          234299 non-null  int64         
     13  Employee_Identifier  234299 non-null  object        
     14  Original_Hire_Date   234299 non-null  datetime64[ns]
    dtypes: datetime64[ns](1), float64(2), int64(4), object(8)
    memory usage: 26.8+ MB
    


```python
# Let's also drop duplicates as our last step in this section

data.drop_duplicates(inplace=True)
```

### Part 3 - Exploratory Data Analysis


```python
# Lets take a look at our dataset

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, notice that we have a feature (column) in our dataset called "Employee_identifier".
# This feature contains unique identification numbers of the employees.
# We can use this feature to determine the number of employees we have in our dataset

data.Employee_Identifier.nunique()
```




    28375




```python
# We know that this dataset contains information about 28375 employees.
# Now let's once again take a look at the number of instances in our dataset

len(data)
```




    234298




```python
# Notice that our dataset contains 234298 instances.
# Let's filter our dataset and select only those instances that contain duplicates of employees identification number.
# Performing this operation we will be only selecting the employees who have repeated instances.

data[data.Employee_Identifier.duplicated() == True]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>2</td>
      <td>2017Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>24473.38</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>234294</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>NORMA</td>
      <td>AMOFA-LOZA</td>
      <td>E.</td>
      <td>STATES ATTORNEY</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>19001.95</td>
      <td>9510048</td>
      <td>ac4574b1-d301-44a7-a220-d3aafb5e9ec3</td>
      <td>2013-01-28</td>
    </tr>
    <tr>
      <th>234295</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>FRANK</td>
      <td>PHILLIPS</td>
      <td>J</td>
      <td>DEPT. OF FACILITIES/MGMT</td>
      <td>1200.0</td>
      <td>DEPT. OF FACILITIES/MGMT</td>
      <td>2452</td>
      <td>Operating Engineer II</td>
      <td>27098.40</td>
      <td>9502511</td>
      <td>9839f658-99a7-4b0c-a338-411d2fc2e8c9</td>
      <td>1988-01-05</td>
    </tr>
    <tr>
      <th>234296</th>
      <td>2018</td>
      <td>1</td>
      <td>2018Q1</td>
      <td>CYNTHIA</td>
      <td>KENDRICK</td>
      <td>R</td>
      <td>ADULT PROBATION DEPT.</td>
      <td>1280.0</td>
      <td>ADULT PROBATION DEPT.</td>
      <td>1567</td>
      <td>Adult Probation Officer- PSB</td>
      <td>19217.40</td>
      <td>9512421</td>
      <td>709a5fed-c9a2-4fe6-9d4d-439bd389b5b8</td>
      <td>1985-07-01</td>
    </tr>
    <tr>
      <th>234297</th>
      <td>2018</td>
      <td>1</td>
      <td>2018Q1</td>
      <td>LAURA</td>
      <td>TOLEDO</td>
      <td>R</td>
      <td>AMBULATORY COMMUNITY HLTH NTWK</td>
      <td>4893.0</td>
      <td>AMBULATORY COMMUNITY HLTH NTWK</td>
      <td>5296</td>
      <td>Medical Assistant</td>
      <td>9698.80</td>
      <td>9519643</td>
      <td>8967b2f6-c6eb-40b6-836b-f52c6954325c</td>
      <td>2016-08-08</td>
    </tr>
    <tr>
      <th>234298</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>NAKIA</td>
      <td>WASHINGTON</td>
      <td>R</td>
      <td>JUVENILE TEMPORARY DETENT.CNTR</td>
      <td>1440.0</td>
      <td>JUVENILE TEMPORARY DETENT.CNTR</td>
      <td>48</td>
      <td>Administrative Assistant III</td>
      <td>12872.44</td>
      <td>1400088</td>
      <td>286c35d5-ca24-4ea7-bb7f-39dff92f78a4</td>
      <td>2011-06-06</td>
    </tr>
  </tbody>
</table>
<p>205923 rows × 15 columns</p>
</div>




```python
# It is clear from the above result that the data is only for 28375 employees,
# but they all have multiple instances that add upto 205923.
# Now upon close inspection, we can also notice that the data is spread across multiple fiscal periods of the years.
# Our goal is to determine the salary of an employee and we need to figure out the base pay per year.
# This information is not available in our dataset, instead it is broken down into multiple quarters of the year.
# So in this case, let's introduce a new feature (column) called "year".
# We can extract the year from the original_hire_date feature and add it to our new feature.

data['year'] = data['Original_Hire_Date'].dt.year
```


```python
# Now let's check if it worked before we move forward.

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now let's target the "base_pay" feature. First, We will group our dataset by the "year" feature.
# Then, we will calculate the average of the "base_pay" feature and plot it on a bar diagram for better understanding.
# Also, we will display only the largest 10 and smallest 10 results only.

# import matplotlib.pyplot as plt

data.groupby('year')['Base_Pay'].mean().nlargest(10).plot(kind = 'bar')
```




    <AxesSubplot:xlabel='year'>




    
![png](output_25_1.png)
    



```python
data.groupby('year')['Base_Pay'].mean().nsmallest(10).plot(kind = 'bar')
```




    <AxesSubplot:xlabel='year'>




    
![png](output_26_1.png)
    



```python
# From the above two cells, we can see the largest 10 and smallest 10 results,
# which are nothing but the highest and lowest average values of the base pay.
# Now let's sort our "year" column.

data['year'].sort_values(ascending=True)
```




    57658     1958
    189805    1958
    57657     1958
    203897    1958
    57661     1958
              ... 
    198591    2018
    207386    2018
    233273    2018
    224078    2018
    226393    2018
    Name: year, Length: 234298, dtype: int64




```python
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
```


    
![png](output_28_0.png)
    



```python
# We can notice which decade had the highest and lowest average base pay from the above result.
# Now let's find out which job titles are paid the highest and which ones are paid the lowest.
# First, let's see how many unique jobs are present in our dataset

data.Job_Title.nunique()
```




    2382




```python
# There are 2382 number of unique jobs.
# Now, using the same logic we used earlier, let's calculate the average base pay according to the job titles.
# We will calculate for both highest and lowest.

data.groupby('Job_Title')['Base_Pay'].mean().nlargest(20)
```




    Job_Title
    Chief Executive Officer -CCHHS    129370.950000
    Senior Director of HIV Service    119270.830000
    Chr of the Div of Urology Surg    118427.760000
    Chair of the Dept of Medicine     112886.584000
    Chair Dept of Trauma  Burn Svc    112265.651000
    Chr of the Div of Neuro Surg      105663.895556
    Ch of the Div of Ad Cardil Cl     104605.001250
    Chr of the Div of Ortho Surg      104058.626000
    Med Dept Chair Surgery            102658.094667
    Ch of the Div of Mat Fet Med      102583.065000
    Chair of Div of Cardioth Surg     101539.677000
    Med Dept Chair Emerg Medicine      96497.792000
    Chief Medical Officer              96016.280000
    Chr of the Div of Otol Surg        94913.097000
    Associated Medical Chairman        94821.650000
    Med Dept Chair - OB GYN            94669.542727
    Med Dept Ch Ortho and Reg Anes     93630.172000
    Ch of the Div of Neuro Anesth      92767.935000
    Ch of the Div of Pain Mgmt         92561.297000
    Chr of the Div of Opht Surg        92236.919000
    Name: Base_Pay, dtype: float64




```python
data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)
```




    Job_Title
    Attendant Patient Care As Req       0.000000
    CADD Operator I                     0.000000
    Clinical Specialist                 0.000000
    Credit Counselor                    0.000000
    Engineering Technician I            0.000000
    Financial Ops Coord - Assessor      0.000000
    Med Dep Chair-Pediatrics            0.000000
    Medical Admin-Ambulatory            0.000000
    Medical Dep Assoc Chair Radio       0.000000
    Medical Dir-Outpatient Svcs         0.000000
    Medical Div Chair-Dir of CCU        0.000000
    Medical Division Chairman VI        0.000000
    Right of Way Agent II               0.000000
    Supply Coordinator - Assessor       0.000000
    Tax Examiner III                    0.000000
    Taxpayer Advocate Analyst IV        0.000000
    Tech Review Specialist IV           0.000000
    Transporter OFH ARNTE               0.000000
    Investigator II Day Report         67.215000
    Associate Judge Circuit Court     150.603296
    Name: Base_Pay, dtype: float64




```python
# The results for the average highest base pay is fine as can be seen above.
# But there is a problem in the average lowest base pay.
# The average of some job titles is 0.
# This is a problem. Maybe the dataset do not contain the "base_pay" information for these job titles.
# Let's fix this issue.

# First let's see how many instances we have with 0 base pay.

data[data.Base_Pay == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>FRANK</td>
      <td>ABBOTT</td>
      <td>J</td>
      <td>Bureau of Admin.</td>
      <td>1500.0</td>
      <td>DEPT. OF TRANSPORTATION AND HW</td>
      <td>431</td>
      <td>Right of Way Agent II</td>
      <td>0.0</td>
      <td>9517610</td>
      <td>f52eb088-72d8-494a-a296-772a71783234</td>
      <td>1986-06-20</td>
      <td>1986</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>KHALIL</td>
      <td>ABBOUD</td>
      <td>E</td>
      <td>Chief Judge</td>
      <td>1280.0</td>
      <td>ADULT PROBATION DEPT.</td>
      <td>1564</td>
      <td>Supervisor Adult Probation</td>
      <td>0.0</td>
      <td>9512676</td>
      <td>e9a92ee0-120f-41f3-b824-4e4d7c43cc5c</td>
      <td>1982-12-17</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>KHALIL</td>
      <td>ABBOUD</td>
      <td>E</td>
      <td>Chief Judge</td>
      <td>1280.0</td>
      <td>ADULT PROBATION DEPT.</td>
      <td>1564</td>
      <td>Supervisor Adult Probation</td>
      <td>0.0</td>
      <td>9512676</td>
      <td>e9a92ee0-120f-41f3-b824-4e4d7c43cc5c</td>
      <td>1982-12-17</td>
      <td>1982</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>HERAND</td>
      <td>ABCARIAN</td>
      <td>E</td>
      <td>Bureau of Health</td>
      <td>4897.0</td>
      <td>STROGER HOSPITAL OF COOK CNTY</td>
      <td>1649</td>
      <td>Medical Div Chairman XII</td>
      <td>0.0</td>
      <td>800768</td>
      <td>e07ba440-b785-4047-ab96-38c9714ce186</td>
      <td>2009-08-17</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2017</td>
      <td>3</td>
      <td>2017Q3</td>
      <td>ABDALLAH</td>
      <td>ABDELHAMID</td>
      <td>F</td>
      <td>Sheriff</td>
      <td>1239.0</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1360</td>
      <td>Correctional Officer</td>
      <td>0.0</td>
      <td>800096</td>
      <td>433c0c09-b58e-45c0-96ee-1cf5c315cf5f</td>
      <td>2011-06-27</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229649</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>THOMAS</td>
      <td>BONDI</td>
      <td>R</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1239.0</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1360</td>
      <td>Correctional Officer</td>
      <td>0.0</td>
      <td>9506963</td>
      <td>272c860d-26bf-44f1-96af-3056e4c654c0</td>
      <td>2012-02-27</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>230477</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>LILIANNA</td>
      <td>KALIN</td>
      <td>M</td>
      <td>HEALTH</td>
      <td>1890.0</td>
      <td>Bureau of Health</td>
      <td>8089</td>
      <td>Sr. Labor &amp; Employ Counsel</td>
      <td>0.0</td>
      <td>1801745</td>
      <td>b12f72e8-e349-42d3-9a36-a3bc62dcabc4</td>
      <td>2006-05-30</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>231487</th>
      <td>2018</td>
      <td>2</td>
      <td>2018Q2</td>
      <td>RAUL</td>
      <td>BARRAZA</td>
      <td>S</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1239.0</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1360</td>
      <td>Correctional Officer</td>
      <td>0.0</td>
      <td>9508604</td>
      <td>76bb6a7e-c9f8-4752-be7d-b80a568525a8</td>
      <td>2012-02-27</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>233666</th>
      <td>2018</td>
      <td>1</td>
      <td>2018Q1</td>
      <td>TANESHA</td>
      <td>WHITE</td>
      <td>L</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1239.0</td>
      <td>DEPARTMENT OF CORRECTIONS</td>
      <td>1360</td>
      <td>Correctional Officer</td>
      <td>0.0</td>
      <td>9508098</td>
      <td>58a40df7-f29b-4141-9c52-a08f49e17f14</td>
      <td>2002-11-18</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>234147</th>
      <td>2018</td>
      <td>1</td>
      <td>2018Q1</td>
      <td>MICHAEL</td>
      <td>JONES</td>
      <td>D</td>
      <td>STROGER HOSPITAL OF COOK CNTY</td>
      <td>4897.0</td>
      <td>STROGER HOSPITAL OF COOK CNTY</td>
      <td>2417</td>
      <td>Hospital Security Officer I</td>
      <td>0.0</td>
      <td>9520507</td>
      <td>9e68518e-c40b-44ec-be2a-0a50d7230e3d</td>
      <td>1990-02-15</td>
      <td>1990</td>
    </tr>
  </tbody>
</table>
<p>2541 rows × 16 columns</p>
</div>




```python
# There are 2541 instances with no information about the base_pay
# Let's remove these rows from our dataframe.
# There are some different ways to do so but the one I find easy to understand is:

data = data[data.Base_Pay != 0]

# In the above line, we are extracting the instances whose base_pay feature does not have 0 
# and adding them back into the same dataframe.
```


```python
# Let's check the number of instances now.

data.shape
```




    (231757, 16)




```python
# Let's repeat the previous steps.

data.Job_Title.nunique()
```




    2364




```python
data.groupby('Job_Title')['Base_Pay'].mean().nsmallest(20)
```




    Job_Title
    Associate Judge Circuit Court      150.844069
    Judge of the Circuit Court         151.700708
    Investigator II Day Report         268.860000
    Manager of Compliance-Revenue      309.110000
    Manager Field Evaluations          334.740000
    Safety Liaison II                  336.095000
    Human Resources Assistant          350.220000
    Dir of Hospitality Services        807.700000
    Traffic Crossing Guards           1159.660647
    Laboratory Technician I           1453.036250
    Student Intern                    1516.666667
    Training Coordinator III          1812.880000
    Deputy Press Secretary            2246.400000
    Hospital Administration Fellow    2767.175556
    Clerk IV                          2774.060388
    Janitor I                         2931.849000
    Intelligence Manager              2985.620000
    Legislative Affairs Adm-Sher      3249.310000
    Front End Developer               3334.620000
    Facilities Liaison-Sheriff        3398.540000
    Name: Base_Pay, dtype: float64




```python
# We have fixed this issue successfully.
# Now, # lets see how the yearly payment is divided in quarters.
# First, we will find out how many times each value in the "fiscal_quarter" feature occurs.

data.Fiscal_Quarter.value_counts()
```




    1    72174
    2    71331
    4    44330
    3    43922
    Name: Fiscal_Quarter, dtype: int64




```python
# In our dataset, the base pay of the employees is spread across fiscal quarters.
# So, let's determine the average base pay for each of these fical quarters.

average_per_quarter = data.groupby('Fiscal_Quarter')['Base_Pay'].mean()
average_per_quarter
```




    Fiscal_Quarter
    1    15223.171414
    2    17872.825494
    3    16411.799485
    4    18965.311050
    Name: Base_Pay, dtype: float64




```python
# Let us generate a pie chart for better understanding.

label = ['Quarter_1','Quarter_2','Quarter_3','Quarter_4' ] 

plt.figure(figsize=(10,6))
plt.pie(average_per_quarter, labels=label, startangle=90)
```




    ([<matplotlib.patches.Wedge at 0x2401a061970>,
      <matplotlib.patches.Wedge at 0x2401a061eb0>,
      <matplotlib.patches.Wedge at 0x2401a0703d0>,
      <matplotlib.patches.Wedge at 0x2401a0708b0>],
     [Text(-0.707334067771227, 0.8424241903994741, 'Quarter_1'),
      Text(-0.878268558492273, -0.6623023019467054, 'Quarter_2'),
      Text(0.6642377272862052, -0.8768057034769202, 'Quarter_3'),
      Text(0.8408621374415534, 0.7091902888627442, 'Quarter_4')])




    
![png](output_39_1.png)
    



```python
# It is easily comprehensible that the base pay is almost equal in all the fiscal quarters.
# Now, for the next step let's repeat the same steps and try to figure out similar statistics for each month.

# To do so, we will first introduce a column that shows the month of hire.
# This step is similar to introducing the "year" column.

data['Month_of_hire'] = data['Original_Hire_Date'].dt.month
```


```python
# We know that the "month_of_hire" column contains the date in numeric format.
# Therefore, we can assume that the new column we just created holds numeric values.
# We can check by calling the head() method.

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
      <th>Month_of_hire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's alter the month_of_hire column to show the name of the month instead of the chronogical number of that month.

data['Month_of_hire'] = pd.to_datetime(data['Month_of_hire'], format='%m').dt.strftime('%B')
```


```python
# Let's check to see if it worked.

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
      <th>Month_of_hire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, let's introduce one more column.
# In this new column we will calculate and store the monthly base pay of the employee.

data['Monthly_Pay'] = data['Base_Pay'] /12
```


```python
# Now, let's verify the changes again.

data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
      <th>Month_of_hire</th>
      <th>Monthly_Pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1953.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1701.901667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1992.066667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1728.816667</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, we can find out in which month most of the people are hired.
# By determining this, we can find out more about the peak hiring season.

# First, let's count the number of times the month_of_hire values are repeated.
# This will give us an idea about how many people are hired in those months.

data.Month_of_hire.value_counts()
```




    June         28287
    July         27355
    August       21280
    April        19693
    September    19060
    May          18745
    October      18021
    February     16897
    March        16413
    November     16236
    December     14978
    January      14792
    Name: Month_of_hire, dtype: int64




```python
# From above result, it is clear that the month of june has the highest number of hires
# and the month of january has the lowest number of hires.
# It is safe to assume that the third quarter of the year is the season where the hiring is at peak.

# Now, let's find out the average base pay for each month.
# Let's also plot a bar chart for better understanding.

data.groupby('Month_of_hire')['Monthly_Pay'].mean().sort_values(ascending = False).plot(kind = 'bar', edgecolor = 'black' )
```




    <AxesSubplot:xlabel='Month_of_hire'>




    
![png](output_47_1.png)
    



```python
# From above visualization, it is clear that the months of July, august and september has the highest average base pay.
# This is the same quarter where the hiring is at peak.
# Can we assume that the new hires in the third quarter are being offered larger base pay packages?
# Let's explore a little more. Let's take a look at other features in our dataset.

data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
      <th>Month_of_hire</th>
      <th>Monthly_Pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1953.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1701.901667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1992.066667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1728.816667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>2</td>
      <td>2017Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>24473.38</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>2039.448333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017</td>
      <td>3</td>
      <td>2017Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>21217.35</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1768.112500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>DAVID</td>
      <td>AARONS</td>
      <td>K</td>
      <td>Assessor</td>
      <td>1040.0</td>
      <td>COUNTY ASSESSOR</td>
      <td>5049</td>
      <td>Residential Model Sr Anal III</td>
      <td>17770.86</td>
      <td>9500731</td>
      <td>f313b1c3-1b1a-4b07-bb75-a8c850a91bac</td>
      <td>1998-09-28</td>
      <td>1998</td>
      <td>September</td>
      <td>1480.905000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>DAVID</td>
      <td>AARONS</td>
      <td>K</td>
      <td>Assessor</td>
      <td>1040.0</td>
      <td>COUNTY ASSESSOR</td>
      <td>5049</td>
      <td>Residential Model Sr Anal III</td>
      <td>20800.67</td>
      <td>9500731</td>
      <td>f313b1c3-1b1a-4b07-bb75-a8c850a91bac</td>
      <td>1998-09-28</td>
      <td>1998</td>
      <td>September</td>
      <td>1733.389167</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>DAVID</td>
      <td>AARONS</td>
      <td>K</td>
      <td>Assessor</td>
      <td>1040.0</td>
      <td>COUNTY ASSESSOR</td>
      <td>5049</td>
      <td>Residential Model Sr Anal III</td>
      <td>17873.76</td>
      <td>9500731</td>
      <td>f313b1c3-1b1a-4b07-bb75-a8c850a91bac</td>
      <td>1998-09-28</td>
      <td>1998</td>
      <td>September</td>
      <td>1489.480000</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Office</th>
      <th>Base_Pay</th>
    </tr>
    <tr>
      <th>Bureau</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bureau of Health</th>
      <td>52886</td>
      <td>19422.269489</td>
    </tr>
    <tr>
      <th>Sheriff</th>
      <td>51311</td>
      <td>16774.608702</td>
    </tr>
    <tr>
      <th>Chief Judge</th>
      <td>24035</td>
      <td>14336.584002</td>
    </tr>
    <tr>
      <th>Clerk of Circuit Ct.</th>
      <td>12891</td>
      <td>13245.250423</td>
    </tr>
    <tr>
      <th>State's Attorney</th>
      <td>11004</td>
      <td>19149.285550</td>
    </tr>
    <tr>
      <th>STROGER HOSPITAL OF COOK CNTY</th>
      <td>7600</td>
      <td>21540.372201</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF CORRECTIONS</th>
      <td>7591</td>
      <td>14002.515391</td>
    </tr>
    <tr>
      <th>Public Defender</th>
      <td>5386</td>
      <td>22071.640046</td>
    </tr>
    <tr>
      <th>CORPORATE</th>
      <td>4929</td>
      <td>11704.547271</td>
    </tr>
    <tr>
      <th>Bureau of Admin.</th>
      <td>4040</td>
      <td>18159.505309</td>
    </tr>
    <tr>
      <th>Facilities Management</th>
      <td>3906</td>
      <td>18212.176086</td>
    </tr>
    <tr>
      <th>HEALTH</th>
      <td>3634</td>
      <td>15373.923621</td>
    </tr>
    <tr>
      <th>CLERK OF CRCT CRT OFF. OF CLER</th>
      <td>3588</td>
      <td>11124.065022</td>
    </tr>
    <tr>
      <th>County Clerk</th>
      <td>3390</td>
      <td>10573.939802</td>
    </tr>
    <tr>
      <th>STATES ATTORNEY</th>
      <td>2421</td>
      <td>19020.710021</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The above result shows the number of offices for each department and their respective average base pay.
# Similarly, we can keep on going and explore the data in many other ways.
# We can use other features as the center of attraction as we did with "base_pay" column and perform similar operations.
# In this way, we are open to several possibilities of exploring the dataset.
# But, our goal is to build a model that predicts the "base_pay". So we kept our operations around that column.

# Let's generate a histogram to check for the distribution of the base_pay column.

plt.figure(figsize=(12,8))
sns.distplot(data['Base_Pay'], color='red')
```




    <AxesSubplot:xlabel='Base_Pay', ylabel='Density'>




    
![png](output_50_1.png)
    



```python
# The histogram provides a visual representation of the distribution of base pay values.
# We can see how frequently different ranges of base pay occurs in the dataset.

# Now let's also check for outliers before moving to next section.

sns.boxplot(data['Base_Pay'])
```




    <AxesSubplot:xlabel='Base_Pay'>




    
![png](output_51_1.png)
    



```python
# It is clearly visible that our dataset does contain some outliers.
# But in my opinion, they seem to be negligeable.
# So, let's conclude our exploratory data analysis section here and move forward without worrying about outliers.

# Notice that we made a few of changes to the dataset.
# We explored and introduced two more columns.
```

### Part 4 - Feature Engineering


```python
# Let's start by creating an extra copy of our dataset.
# We will do so in case we make changes that can't be reversed.

data2 = data
```


```python
# Verify the previous step

data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Fiscal_Period</th>
      <th>First_Name</th>
      <th>Last_Name</th>
      <th>Middle_Init</th>
      <th>Bureau</th>
      <th>Office</th>
      <th>Office_Name</th>
      <th>Job_Code</th>
      <th>Job_Title</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>Employee_Identifier</th>
      <th>Original_Hire_Date</th>
      <th>year</th>
      <th>Month_of_hire</th>
      <th>Monthly_Pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>2016Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>2016Q2</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1953.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>2016Q3</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1701.901667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>2016Q4</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1992.066667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2017Q1</td>
      <td>AMRITH</td>
      <td>AAKRE</td>
      <td>K</td>
      <td>State's Attorney</td>
      <td>1250.0</td>
      <td>STATES ATTORNEY</td>
      <td>1172</td>
      <td>Assistant State's Attorney</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>6ac7ba3e-d286-44f5-87a0-191dc415e23c</td>
      <td>2005-05-16</td>
      <td>2005</td>
      <td>May</td>
      <td>1728.816667</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, let's start by dropping the columns that are not relevant to our goal.
# Let's drop irrelevant columns and keep only those that are relevant.
# We can do so by using the drop method. But I think there is a more eaier way to do it.
# We can simply select and copy the columns we need from the original dataset.

data2 = data[['Fiscal_Year', 'Fiscal_Quarter', 'Office','Job_Code', 'Base_Pay', 'Position_ID', 'year','Month_of_hire', 'Monthly_Pay']]
data2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Office</th>
      <th>Job_Code</th>
      <th>Base_Pay</th>
      <th>Position_ID</th>
      <th>year</th>
      <th>Month_of_hire</th>
      <th>Monthly_Pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>20088.00</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>23436.00</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1953.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>20422.82</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1701.901667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>23904.80</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1992.066667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>20745.80</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1728.816667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>234294</th>
      <td>2018</td>
      <td>2</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>19001.95</td>
      <td>9510048</td>
      <td>2013</td>
      <td>January</td>
      <td>1583.495833</td>
    </tr>
    <tr>
      <th>234295</th>
      <td>2018</td>
      <td>2</td>
      <td>1200.0</td>
      <td>2452</td>
      <td>27098.40</td>
      <td>9502511</td>
      <td>1988</td>
      <td>January</td>
      <td>2258.200000</td>
    </tr>
    <tr>
      <th>234296</th>
      <td>2018</td>
      <td>1</td>
      <td>1280.0</td>
      <td>1567</td>
      <td>19217.40</td>
      <td>9512421</td>
      <td>1985</td>
      <td>July</td>
      <td>1601.450000</td>
    </tr>
    <tr>
      <th>234297</th>
      <td>2018</td>
      <td>1</td>
      <td>4893.0</td>
      <td>5296</td>
      <td>9698.80</td>
      <td>9519643</td>
      <td>2016</td>
      <td>August</td>
      <td>808.233333</td>
    </tr>
    <tr>
      <th>234298</th>
      <td>2018</td>
      <td>2</td>
      <td>1440.0</td>
      <td>48</td>
      <td>12872.44</td>
      <td>1400088</td>
      <td>2011</td>
      <td>June</td>
      <td>1072.703333</td>
    </tr>
  </tbody>
</table>
<p>231757 rows × 9 columns</p>
</div>




```python
# Now, let's check for collinearity between the columns in our new dataset

plt.figure(figsize=(12,10))
sns.heatmap(data2.corr(), linecolor='black', linewidths=2.0, annot=True, cmap='Blues_r')
plt.xticks(fontsize = 12)
```




    (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
     [Text(0.5, 0, 'Fiscal_Year'),
      Text(1.5, 0, 'Fiscal_Quarter'),
      Text(2.5, 0, 'Office'),
      Text(3.5, 0, 'Job_Code'),
      Text(4.5, 0, 'Base_Pay'),
      Text(5.5, 0, 'Position_ID'),
      Text(6.5, 0, 'year'),
      Text(7.5, 0, 'Monthly_Pay')])




    
![png](output_57_1.png)
    



```python
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
```




    (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
     [Text(0.5, 0, 'Fiscal_Year'),
      Text(1.5, 0, 'Fiscal_Quarter'),
      Text(2.5, 0, 'Office'),
      Text(3.5, 0, 'Job_Code'),
      Text(4.5, 0, 'Position_ID'),
      Text(5.5, 0, 'year'),
      Text(6.5, 0, 'Monthly_Pay')])




    
![png](output_58_1.png)
    



```python
# Now, there is no such heavy collinearity in our dataset.
# Also, we separated our target value from the dataset so from here on we will use the x dataframe.
# Let's check our dataframe.

x.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Office</th>
      <th>Job_Code</th>
      <th>Position_ID</th>
      <th>year</th>
      <th>Month_of_hire</th>
      <th>Monthly_Pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1953.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1701.901667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1992.066667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>May</td>
      <td>1728.816667</td>
    </tr>
  </tbody>
</table>
</div>




```python
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 231757 entries, 0 to 234298
    Data columns (total 8 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   Fiscal_Year     231757 non-null  int64  
     1   Fiscal_Quarter  231757 non-null  int64  
     2   Office          231757 non-null  float64
     3   Job_Code        231757 non-null  int64  
     4   Position_ID     231757 non-null  int64  
     5   year            231757 non-null  int64  
     6   Month_of_hire   231757 non-null  object 
     7   Monthly_Pay     231757 non-null  float64
    dtypes: float64(2), int64(5), object(1)
    memory usage: 15.9+ MB
    


```python
# Here, notice that the month_of_hire is a categorical column which includes 12 categories (Months of the year).
# We can break this column down into 12 separate columns and use each of them as a boolean type instead of object string.
# First we will create a dataframe and extract each of the value from "month_of_hire" column.
# We will insert these values into new dataframe as columns.

x_dummies = pd.get_dummies(data['Month_of_hire'], prefix = 'Month',prefix_sep='_',drop_first=True)
x_dummies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month_August</th>
      <th>Month_December</th>
      <th>Month_February</th>
      <th>Month_January</th>
      <th>Month_July</th>
      <th>Month_June</th>
      <th>Month_March</th>
      <th>Month_May</th>
      <th>Month_November</th>
      <th>Month_October</th>
      <th>Month_September</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>234294</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>234295</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>234296</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>234297</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>234298</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>231757 rows × 11 columns</p>
</div>




```python
# Now let's join this dataframe to our dataframe and drop the "month_of_hire" column.

x = x.join(x_dummies)
x.drop(columns=['Month_of_hire'], inplace=True)
```


```python
x.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fiscal_Year</th>
      <th>Fiscal_Quarter</th>
      <th>Office</th>
      <th>Job_Code</th>
      <th>Position_ID</th>
      <th>year</th>
      <th>Monthly_Pay</th>
      <th>Month_August</th>
      <th>Month_December</th>
      <th>Month_February</th>
      <th>Month_January</th>
      <th>Month_July</th>
      <th>Month_June</th>
      <th>Month_March</th>
      <th>Month_May</th>
      <th>Month_November</th>
      <th>Month_October</th>
      <th>Month_September</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>1674.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>2</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>1953.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>3</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>1701.901667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>4</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>1992.066667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1250.0</td>
      <td>1172</td>
      <td>9510200</td>
      <td>2005</td>
      <td>1728.816667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 231757 entries, 0 to 234298
    Data columns (total 18 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   Fiscal_Year      231757 non-null  int64  
     1   Fiscal_Quarter   231757 non-null  int64  
     2   Office           231757 non-null  float64
     3   Job_Code         231757 non-null  int64  
     4   Position_ID      231757 non-null  int64  
     5   year             231757 non-null  int64  
     6   Monthly_Pay      231757 non-null  float64
     7   Month_August     231757 non-null  uint8  
     8   Month_December   231757 non-null  uint8  
     9   Month_February   231757 non-null  uint8  
     10  Month_January    231757 non-null  uint8  
     11  Month_July       231757 non-null  uint8  
     12  Month_June       231757 non-null  uint8  
     13  Month_March      231757 non-null  uint8  
     14  Month_May        231757 non-null  uint8  
     15  Month_November   231757 non-null  uint8  
     16  Month_October    231757 non-null  uint8  
     17  Month_September  231757 non-null  uint8  
    dtypes: float64(2), int64(5), uint8(11)
    memory usage: 24.6 MB
    


```python
# Notice that we created 12 new columns and each of these columns has an int datatype with 8 bits.
# These new columns contain two values 0 and 1.
# 0 meaning False and 1 meaning True.
# we eliminated the object string and made it a little easier to perform operations.

# Now let's standardize the dataset.

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# The above line creates a new instance of StandardScaler Class.
# The StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
```


```python
# Let's standardize the features stored in x

x[x.columns] = scaler.fit_transform(x[x.columns])

# Let me explain the above line. Here's what happens:
# First, the 'x.columns' returns the column names of x.
# Then, 'x[x.columns]' selects all the columns in x.
# After that, the 'scaler.fit_transform(x[x.columns])' part scales all the selected columns.
# The resulting scaled values are then assigned back to the respective columns in x using the same 'x[x.columns]'.
```


```python
# Let's reshape the y dataframe.

y = y.values.reshape(-1,1)

# The above line will convert the single column in the y dataframe into a 2D array.
```


```python
# Let's scale the values in y dataframe as well.
# This way we will ensure that the target values are also scaled to unit values.

y = scaler.fit_transform(y)
```


```python
# By performing the previous steps, we eliminated any chance of biases among the data in our columns.
# All the columns noiw contain unit values.
# Let's move to the next step which is to build a predictive model.
```

### Part 5 - Predictive Modeling and Validation


```python
# First we will need to split our dataset into training data and testing data.

# from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.75, random_state=42,shuffle=True)

# Here we extracted 75% of data randomly and created a training datset with it.
# The remaining 25% data will automatically be assigned as testing data.
```


```python
# Let's check the 4 dataframes we just created

x_train.shape
```




    (173817, 18)




```python
x_test.shape
```




    (57940, 18)




```python
y_train.shape
```




    (173817, 1)




```python
y_test.shape
```




    (57940, 1)




```python
# Now that our training data and testing data is ready, Let's proceed to building a model.
# First let's perform Linear Regression.

# from sklearn.linear_model import LinearRegression

# Creating an model instance of the LinearRegression class.

Linear_model = LinearRegression()

# Fitting the model with the training data.

Linear_model.fit(x_train, y_train)

# Let's use the predict method to make the model predict our target values.

y_pred_linear = Linear_model.predict(x_test)
```


```python
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
```

    The R-Squared Score is: 1.0
    The Mean Squared Error is: 1.0710553941627577e-29
    The Root Mean Squared Error is: 3.2726982662059723e-15
    The Mean Absolute Error is: 2.3121412355791717e-15
    


```python
# Let's Validate by comparing the R-Squared score of the model's predicted data to the actual data.

r2_train = r2_score(y_train, Linear_model.predict(x_train))
r2_train
```




    1.0




```python
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_linear)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
```


    
![png](output_79_0.png)
    



```python
# Let's repeat the above steps but with a different model.
# Let's experiment with Support Vector Regressor

# from sklearn.svm import SVR

model_svr = SVR()

model_svr.fit(x_train,y_train)
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVR()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR()</pre></div></div></div></div></div>




```python
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
```

    The R-Squared Score is: 0.9906385016760155
    The Mean Squared Error is: 0.00923342090853018
    The Root Mean Squared Error is: 0.09609069106073793
    The Mean Absolute Error is: 0.03511641023178011
     
    Validating SVR: 
    




    0.9927326384950533




```python
# Here, the R2 score of training dataset is slightly more than that of testing dataset which means that model is performing accurately.
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_svr)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
```


    
![png](output_82_0.png)
    



```python
# Let's repeat the above steps again but with a different model.
# Let's experiment with Decision Tree Regressor

# from sklearn.tree import DecisionTreeRegressor

model_DTR = DecisionTreeRegressor()

model_DTR.fit(x_train,y_train)
```




<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor()</pre></div></div></div></div></div>




```python
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
```

    The R-Squared Score is: 0.9999858075675555
    The Mean Squared Error is: 1.3998261596707264e-05
    The Root Mean Squared Error is: 0.003741425075650622
    The Mean Absolute Error is: 0.00010082256125855401
     
    Validating DTR: 
    




    1.0




```python
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_DTR)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
```


    
![png](output_85_0.png)
    



```python
# Let's repeat the above steps but with a different model.
# Let's experiment with Random Forest Regression.

# from sklearn.ensemble import RandomForestRegressor

model_RF = RandomForestRegressor()

model_RF.fit(x_train,y_train)
```




<style>#sk-container-id-7 {color: black;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div>




```python
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
```

    The R-Squared Score is: 0.9999843125198861
    The Mean Squared Error is: 1.5472855078661303e-05
    The Root Mean Squared Error is: 0.0039335550178764885
    The Mean Absolute Error is: 8.850695970900948e-05
     
    Validating Random Forest: 
    




    0.9999777611635601




```python
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_RF)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
```


    
![png](output_88_0.png)
    


### Part 6 - Optimization


```python
# Notice that out of all the models we have trained, Random Forest has the lowest MSE with corresponding R2 score.
# Therefore, Let's Optimize Random Forest Regression model.
# Let's use GridSearchCV for Model Tuning.

# from sklearn.model_selection import GridSearchCV
# from sklearn import ensemble

model_t = GridSearchCV(ensemble.RandomForestRegressor(), {'n_estimators': [10,15, 30], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 4],},cv=3)
model_t.fit(x_train,y_train)
```




<style>#sk-container-id-8 {color: black;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3, estimator=RandomForestRegressor(),
             param_grid={&#x27;max_depth&#x27;: [5, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;n_estimators&#x27;: [10, 15, 30]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3, estimator=RandomForestRegressor(),
             param_grid={&#x27;max_depth&#x27;: [5, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;n_estimators&#x27;: [10, 15, 30]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div></div></div></div>




```python
# Let's breakdown the above two lines to understadn what is being done here.

# 'ensemble.RandomForestRegressor()`: This part creates an instance of a RandomForestRegressor model
# This instance is an ensemble learning method for used regression tasks.

# `{'n_estimators': [10,15, 30], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 4],}`
# This part is a dictionary specifying the hyperparameter grid to search.
# It includes different values for 'n_estimators' (number of trees in the forest), 'max_depth' (maximum depth of the trees), and 'min_samples_leaf' (minimum samples required to be at a leaf node).

# `cv=3` This part specifies the number of folds for cross-validation during the grid search.
# In this case, we've set it to 3, meaning the data will be split into 3 subsets for cross-validation.

# `model_t.fit(x_train, y_train)`: This line fits the GridSearchCV model to the training data:
```


```python
# Now that we have performed the GridSearchCV, Let's print out some information we need.

print(model_t.best_estimator_)
print(model_t.best_params_)
print(model_t.best_score_)
```

    RandomForestRegressor(max_depth=15, n_estimators=15)
    {'max_depth': 15, 'min_samples_leaf': 1, 'n_estimators': 15}
    0.9998230075827242
    


```python
# We have found the best parameters and now we can use these to optimize our model.
# Let's repeat the Random Forest Regression steps with optimized parameters.

final_model = RandomForestRegressor(max_depth=20, n_estimators=30, min_samples_leaf=1)

final_model.fit(x_train,y_train)
```




<style>#sk-container-id-9 {color: black;}#sk-container-id-9 pre{padding: 0;}#sk-container-id-9 div.sk-toggleable {background-color: white;}#sk-container-id-9 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-9 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-9 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-9 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-9 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-9 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-9 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-9 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-9 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-9 div.sk-item {position: relative;z-index: 1;}#sk-container-id-9 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-9 div.sk-item::before, #sk-container-id-9 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-9 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-9 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-9 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-9 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-9 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-9 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-9 div.sk-label-container {text-align: center;}#sk-container-id-9 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-9 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_depth=20, n_estimators=30)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" checked><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(max_depth=20, n_estimators=30)</pre></div></div></div></div></div>




```python
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
```

    The R-Squared Score is: 0.9999865466761145
    The Mean Squared Error is: 1.3269264999496914e-05
    The Root Mean Squared Error is: 0.003642700234646946
    The Mean Absolute Error is: 9.468430089747949e-05
     
    Validating the optimized Random Forest: 
    




    0.999968287488737




```python
# We can visualize these findings to understand it better.

plt.figure(figsize=(10,6))
plt.scatter(y_test,predicted)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
```


    
![png](output_95_0.png)
    



```python
# Notice that it took a really long time for GridSearchCV to train the model.
# Let's save these models so we won't have to repeat the whole process.

# import joblib as jb

# Saving the optimized Random Forest Model:

jb.dump(final_model, filename= r'C:\Users\Amaan Ahmed\Music\Project\RF_model.pkl')

# Saving the Grid Search Model:

jb.dump(model_t, filename=r'C:\Users\Amaan Ahmed\Music\Project\grid_estimator.pkl')
```




    ['C:\\Users\\Amaan Ahmed\\Music\\Project\\grid_estimator.pkl']



# END OF PROJECT

### Thoughts regarding the project outlines and programming practices:

In my opinion, our code looks well-organized and demonstrates a solid understanding of data science and machine learning principles, not to mention we've also followed good programming practices. Let's discuss some of the positive aspects to learn from this project:

1. Modular Structure: We've broken down our code into different sections, which makes it easy to read and understand. Each section has a clear purpose and is appropriately commented.

2. Clear and Informative Comments: Our comments provide helpful explanations of what each section of code does. This is very important for anyone reading your code, including yourself in the future.

3. Consistent Naming Conventions: Our variable names are clear and follow a consistent naming convention, which enhances code readability.

4. Use of Libraries: We've imported the necessary libraries at the beginning, which is considered a good practice.

5. Exploratory Data Analysis: We've performed a thorough exploratory data analysis, which is crucial in understanding the dataset before applying any machine learning models.

6. Data Preprocessing: We've handled missing values, converted data types, and standardized the features, which are essential steps in data preprocessing.

7. Modeling and Evaluation: We've applied different machine learning models, evaluated their performance, and even performed model optimization using GridSearchCV.

8. Visualizations: We've used visualizations to present our findings, which is a great way to communicate insights.

9. Optimization: We've gone a step further by optimizing the Random Forest model using GridSearchCV.

10. Validation and Model Comparison: We've validated our models and compared their performance, providing a clear understanding of how well they are performing.

Note: The above comments are my personal opinion and you are not obligated to agree with them.


```python

```
