# Machine Learning Approach for Employee Salary Prediction

**Amaan Ahmed**  
Master of Science in Data Science <br>
Mr.amaan.ahmed@gmail.com

## Abstract

In this study, we delve into the realm of machine learning and data science to predict employee salaries. Our dataset, sourced from open data repositories, 
provides detailed salary information for a specific country and year. Through meticulous preprocessing and analysis, we glean valuable insights.

Beyond analysis, we employ a suite of regression models to forecast base pay for new employees based on dataset features. The results are exceptionally promising, 
demonstrating the potential of this approach for organizations seeking to navigate the dynamic job market and determine competitive compensation packages. 
Among the models investigated, the Random Forest regressor emerges as the most accurate, boasting an impressive R-squared score of 0.99. This model's hyperparameters are 
meticulously fine-tuned using Grid Search techniques, ensuring robust performance within the specified dataset range.

Our project offers a comprehensive framework for understanding and predicting employee compensation, with the Random Forest regressor standing out as a particularly powerful tool in this endeavor.

*Keywords* - *Regression, Machine Learning, Tuning, Analysis, Salary*

## I. INTRODUCTION

Data-driven decision-making is a very powerful method and approach in today's world used to tackle critical situations that require highly precise computation. 
Machine learning is a powerful tool used by organizations to make such decisions. One such application of machine learning is in the field of the Job Market. 
Determining the salary of an employee who is either joining the organization from day 1 or the promoted employee is very important. 
Fair pay is no doubt one of the most important and expected privileges of an employee but at the same time, for the organization, to make 
such a decision can be critical as it requires and is affected by multiple factors. Employee retention and the growth of the organization both are directly dependent on the offered pay of 
the employee and their satisfaction. Thus, it cannot be considered a problem to be ignored. With this study, we shall explore how machine learning and Data science can help both 
the employee and the employer determine the right amount of pay for them.

### ***A. Aim***

The aim of this project is to leverage machine learning and data science methodologies to predict employee salaries based on relevant features within a comprehensive dataset. 
By employing regression models, we seek to provide organizations with a reliable tool for estimating base pay, thereby aiding in effective compensation planning.

### ***B. Objectives***

The following are the important objectives of the project:

1. Data Collection and Preprocessing: This phase involves sourcing the dataset from open data resources and conducting thorough data cleaning, including handling missing values and removing duplicates. The goal is to ensure that the dataset is accurate and suitable for modeling.

2. Exploratory Data Analysis (EDA): Through EDA, we aim to gain a deeper understanding of the dataset's characteristics, uncover patterns, and extract meaningful insights. This step is crucial in identifying key features that influence employee compensation.

3. Feature Engineering: This stage focuses on selecting and transforming relevant attributes to enhance the dataset's suitability for modeling. Techniques such as one-hot encoding and standardization will be applied to refin0e the data.

4. Predictive Modeling and Evaluation: Utilizing regression models including Linear Regression, Support Vector Regression, Decision Tree regression, and Random Forest regression, we aim to predict base pay for new employees. Performance will be evaluated using metrics like R-Squared, Mean Squared Error, Root Mean Squared Error, and Mean Absolute Error.

5. Model Optimization: Through Grid Search techniques, we will fine-tune the Random Forest Regressor to maximize predictive accuracy. This step aims to ensure the model performs optimally within the dataset's specified range.

By achieving these objectives, we endeavor to provide organizations with a robust framework for understanding and forecasting employee compensation, ultimately contributing to informed decision-making in the realm of workforce management.

## II. DATASET DESCRIPTION

The "Employee Payroll" dataset serves as the cornerstone of this project, providing a comprehensive view of employee compensation within a specific context. 
This dataset encompasses a range of attributes, each contributing to a holistic understanding of salary structures and associated factors.

The dataset is sourced from open data resources and focuses on salary information within a single country for the duration of three years. 
This temporal scope enables an in-depth analysis of compensation trends and variations over the course of the specified time frame. The data collection process ensures accuracy and reliability, 
forming a solid foundation for subsequent analysis and modeling.

The dataset undergoes meticulous preprocessing to guarantee data integrity. This includes addressing missing values through forward-fill imputation and eliminating duplicates to ensure a clean, 
representative dataset. Such preprocessing steps are essential for generating accurate insights and building robust predictive models.

_**TABLE I.  	Dataset Attribute Description**_

| Attribute | Type | Example Value |
| --------- | ---- | ------------- |
| Fiscal Year | Integer | 2016 |
| Fiscal Quarter | Integer | 2 |
| Fiscal Period | String | 2016Q2 |
| First Name | String | DAVID |
| Last Name | String | ABBO |
| Middle Initial | String | K |
| Bureau | String | ASSESSOR |
| Office | Integer | 1040 |
| Office Name | String | STATES ATTORNEY |
| Job Code | Integer | 1172 |
| Job Title | String | ASSISTANT STATES ATTORNEY |
| Base Pay | Float | 27852.5 |
| Position ID | Integer | 9510200 |
| Employee Identifier | String | 6ac7ba3e-d286-44f5-87a0-191dc415e23c |
| Original Hire Date | Object | 5/16/2005 |

This table provides an overview of the features present in the dataset, along with their respective data types. Understanding the nature of each attribute is crucial for effective analysis and modeling.

## III. METHODOLOGY

The methodology employed in this project adopts a systematic approach to the analysis and prediction of employee salaries. This process involves several distinct stages, each contributing significantly to the overall goal of accurate salary estimation.

**Data Collection and Preprocessing:**

Data Sourcing: The dataset is meticulously collected from reputable open data resources. It provides a comprehensive snapshot of employee compensation within a specific country for a given year. This ensures that the dataset is representative and suitable for in-depth analysis.

Data Cleaning: Rigorous preprocessing is conducted to ensure data integrity. Missing values are addressed through forward-fill imputation, a method chosen for its suitability in this context. Duplicate entries, if any, are systematically removed. This crucial step lays the foundation for subsequent analysis and modeling, ensuring that the data is reliable and free from inconsistencies.

**Exploratory Data Analysis (EDA):**

Understanding Distributions: The EDA phase aims to gain a comprehensive understanding of the distribution of key variables. Through a combination of visualizations, descriptive statistics, and frequency distributions, we delve into the characteristics of features such as base pay, fiscal year, and other relevant attributes. This insight is pivotal in identifying potential patterns and outliers.

Identifying Patterns: In addition to distributions, the EDA process involves the identification of patterns and trends within the dataset. This step is instrumental in recognizing factors that may influence employee compensation. Scatter plots, box plots, and time series analysis are employed to uncover correlations and temporal trends, providing valuable insights into the underlying dynamics.

**Feature Engineering:**

Feature Selection: Relevant attributes are selected based on their potential impact on salary prediction. This involves careful consideration of factors such as fiscal year, office location, job code, and position ID. Features that demonstrate a significant influence on base pay are prioritized for inclusion in the modeling process.

Data Transformation: Techniques such as one-hot encoding and standardization are applied to refine the dataset. Categorical variables are appropriately encoded to ensure compatibility with regression models. Standardization, a critical step in feature engineering, ensures that all variables are on a comparable scale, preventing any one attribute from dominating the modeling process.

**Predictive Modeling and Evaluation:**

Model Selection: Various regression models are implemented to predict base pay. These include Linear Regression, Support Vector Regression, Decision Tree Regressor, and Random Forest Regressor. Each model is rigorously evaluated using established metrics such as R-squared, Mean Squared Error, Root Mean Squared Error, and Mean Absolute Error. This allows for a comprehensive comparison of their performance and assists in the identification of the most accurate predictor of base pay.

Model Comparison: The performance of each model is compared to determine the most accurate and reliable predictor of base pay. This critical evaluation stage aids in selecting the model that best aligns with the project's objectives and the dataset's characteristics.

**Model Optimization:**

Hyperparameter Tuning: The Random Forest Regressor, identified as the most accurate model, undergoes further refinement through Grid Search techniques. This step fine-tunes hyperparameters, including the number of estimators and maximum depth, to maximize predictive accuracy. By systematically exploring the parameter space, we ensure that the model achieves optimal performance within the specified dataset range.
Conclusion and Recommendations:

Optimized Model Application: The final Random Forest Regressor, equipped with optimized hyperparameters, stands as a powerful tool for accurately estimating employee salaries. Organizations can leverage this model to make informed decisions regarding compensation packages, aiding in talent acquisition and retention strategies.

Future Considerations: Recommendations for future research may include expanding the dataset to encompass multiple countries and years, or exploring additional features that could enhance salary prediction accuracy. Additionally, the model's applicability to other aspects of human resources and workforce management could be explored, offering further avenues for organizational improvement.

## IV. DISCUSSION AND CONCLUSION

The results of this project offer valuable insights into employee compensation and predictive modeling within the specified context. Several key points emerge from the analysis and modeling process:

**Model Performance and Accuracy:**

The regression models employed in this study demonstrated high levels of accuracy in predicting employee salaries. The Random Forest Regressor, in particular, exhibited exceptional performance, achieving an impressive R-Squared score of 0.99. This suggests that the model accounts for a substantial proportion of the variability in base pay, indicating its effectiveness in making accurate predictions.

**Feature Importance:**

Analysis of feature importance revealed significant attributes that strongly influence base pay predictions. Factors such as fiscal year, office location, job code, and position ID emerged as critical drivers of compensation. This information equips organizations with actionable insights for strategic decision-making in salary structuring.

**Practical Applications:**

The developed salary prediction model holds considerable practical applications for organizations. It can serve as a valuable tool in various HR functions, including talent acquisition, workforce planning, and budget allocation. By providing accurate salary estimates, organizations can make informed decisions to attract and retain top talent.

**Limitations and Future Work:**

It is essential to acknowledge the limitations of the project. The dataset's temporal and geographical scope may restrict the model's generalizability to broader contexts. Future work could involve expanding the dataset to include multiple countries and years, thereby enhancing the model's applicability. Additionally, exploring advanced modeling techniques or incorporating additional features could further improve predictive accuracy.

**Organizational Impact:**

The impact of this project on organizational decision-making is substantial. By leveraging accurate salary predictions, organizations can optimize their compensation strategies, aligning them with industry standards and employee expectations. This, in turn, contributes to enhanced employee satisfaction and retention rates.

**Ethical Considerations:**

Ethical implications surrounding salary prediction are of paramount importance. Ensuring fairness and equity in compensation practices is imperative. Organizations must remain vigilant in avoiding potential biases and disparities that may arise from automated salary estimations. Regular audits and transparency in model development can help mitigate ethical concerns.

In conclusion, the results of this project signify a significant step forward in understanding and predicting employee compensation. The high accuracy of the Random Forest Regressor, coupled with the identification of influential features, empowers organizations with a powerful tool for strategic compensation planning. While acknowledging limitations, this study lays the foundation for future research and highlights the potential for impactful contributions to the field of human resources and talent management.

