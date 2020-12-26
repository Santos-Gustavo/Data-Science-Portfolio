# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


human_resources = pd.read_csv('Human_Resources.csv')
human_resources.head()

human_resources.info()

human_resources.drop(['EmployeeCount','StandardHours','EmployeeNumber','Over18'], axis=1, inplace=True)


# Replace 'Attrition' , 'OverTime' , 'gender' column with integers before performing any visualizations 
human_resources['Attrition'] = human_resources['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
human_resources['OverTime'] = human_resources['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

# See if we have any missing data!
sns.heatmap(human_resources.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')

# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy
human_resources.hist(bins=30,
                     figsize=(20,20),
                     color='r')

# Let's see how many employees left the company! 
employees_left = human_resources[human_resources.Attrition == 1]
employees_stayed = human_resources[human_resources.Attrition == 0]

# Count the number of employees who stayed and left
# It seems that we are dealing with an imbalanced dataset 
print('Number of employees who stayed =',len(employees_stayed))
print('% of employees who stayed =', round(len(employees_stayed)/len(human_resources) * 100, 2),'% \n')

print('Number of employees who left =',len(employees_left))
print('% of employees who left =', round(len(employees_left)/len(human_resources) * 100, 2),'%')

employees_left.describe()

#  Compare the mean and std of the employees who stayed and left 
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home 
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level

employees_stayed.describe()

correlation = human_resources.corr()

plt.figure(figsize=(20,12))
sns.heatmap(correlation,
            annot=True, 
            cmap="YlGnBu")
# Job level is strongly correlated with total working years
# Monthly income is strongly correlated with Job level
# Monthly income is strongly correlated with total working hours
# Age is correlated with monthly income

#It seems that younger employees tend more to quit than older ones
# To be sure must see KDE
plt.figure(figsize=[25, 12])
sns.countplot(x = 'Age', hue = 'Attrition', data = human_resources)

plt.figure(figsize=[25, 12])
sns.countplot(x = 'JobRole', hue = 'Attrition', data = human_resources)

# As suspected, the KDE shows that employees younger than 33 years quit more often
plt.figure(figsize=(14,9))
sns.kdeplot(employees_left.Age, label='Employees who left', shade=True, color='r',)#
sns.kdeplot(employees_stayed.Age, label='Employees who stayed', shade=True, color='b')

plt.figure(figsize=(18,12))
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = human_resources,)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = human_resources)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = human_resources)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = human_resources)

# Single employees tend to leave more compared to married and divorced
# Sales Representitives & Laboratory Tecnician tend to leave more compared to any other job 
# Less involved employees tend to leave the company 
# Less experienced (low job level) tend to leave the company

#Employees who live closer tend to remain in the company
#The density changes at approximately 10
plt.figure(figsize=(14,8))

sns.kdeplot(employees_left.DistanceFromHome, label='Employees who left', shade=True, color='r')
sns.kdeplot(employees_stayed.DistanceFromHome, label='Employees who left', shade=True, color='b')

# Sales Representative & Laboratory Technician are bewteen the lowest payments
plt.figure(figsize=(15,12))

sns.boxplot(x='MonthlyIncome',
            y='JobRole',
            data=human_resources)



# To a better result the data must be converted to full Integer
# BusinessTravel, Department, EducationField, JobRole, MaritalStatus must become integers
human_resources.head(2)

# Select the collumns to be converted
X_cat = human_resources[['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']]

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()

# .shape to be sure that everything went fine
X_cat.shape

# Convert into X_cat into a DataFrame to concat with our DataFrame
 X_cat = pd.DataFrame(X_cat)
 X_cat

# Create one DataFrame just with the integers of our initial DataFrame
X_numerical = human_resources[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]

# Concat both DataFrames
X_all = pd.concat([X_numerical,X_cat], axis=1)
X_all.head()

# Notice the scale of the DataFrame isn't balanced, while our DistanceFromHome have values between 1-40, DailyRate have values over 1000
# To have the best performance of our Model we must scale

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X_all)
X

y = human_resources['Attrition']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.shape

X_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_test

from sklearn.metrics import confusion_matrix, classification_report

print('Accuracy {}%'.format(100 * accuracy_score(y_pred, y_test)))

cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Testing Set Performance
cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))