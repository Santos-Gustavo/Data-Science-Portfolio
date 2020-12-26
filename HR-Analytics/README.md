# HR Analytics Employee Attrition & Performance: Project Overview


Hiring and retaining employees are extremely complex tasks that require capital, time and skills.
>
> “Small business owners spend 40% of their working hours on tasks that do not generate any income such as hiring”. 
>
> “Companies spend 15%-20% of the employee's salary to recruit a new candidate”.
>
> “An average company loses anywhere between 1% and 2.5% of their total revenue on the time it takes to bring a new hire up to speed”.
>
> “Hiring a new employee costs an average of $7645 (0-500 corporation)”.
>
> “It takes 52 days on average to fill a position”. 

[Source](https://toggl.com/blog/cost-of-hiring-an-employee)


## Code and Resource Used
**Python Version:** 3.8.3  
**Packages:** Pandas, numpy, Matplotlib, Seaborn, Sklearn  
**Data Source:** https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset  

## Data Cleaning
I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Removed rows with just one value
    * Over18
* Removed rows with personal value
    * EmployeeCount, EmployeeNumber
* Transformed 'Yes' and 'No' values into dummy values
* Created columns for:
    * Employees that stay
    * Employees that left

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights.
![alt text](https://github.com/Santos-Gustavo/Data-Science-Portfolio/blob/main/HR-Analytics/Images/chance_to-quit_by_job_role.jpg)
![alt text](https://github.com/Santos-Gustavo/Data-Science-Portfolio/blob/main/HR-Analytics/Images/distance_from_home.png)
![alt text](https://github.com/Santos-Gustavo/Data-Science-Portfolio/blob/main/HR-Analytics/Images/quits_by_age.png)

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 25%.   

I tried two different models and evaluated them using F1 Score. I chose F1 because it provides a balance between recall and precision in the presence of unbalanced datasets.   

I tried two different models:
*	**Logistic Reggression** – Is used good to predict binary outputs on a continuos spectrum
*	**Random Forest** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective

## Model performance
The Logistic Regression model was slightly better on the test. 
*	**Logistic Reggresion** : F1 = 91%
*	**Random Forest**: F1 = 87%
