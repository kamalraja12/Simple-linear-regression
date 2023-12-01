# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 23:15:15 2023

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

salary=pd.read_csv('Downloads\Salary_Data.csv')
salary

salary.describe()
salary.corr()

x=salary.YearsExperience
y=salary.Salary
plt.scatter(x,y)
plt.xlabel = ('YearsExperience')
plt.ylabel = ('Salary')

sns.distplot(salary['YearsExperience'])
sns.distplot(salary['Salary'])

sns.regplot(x="YearsExperience", y="Salary", data=salary)

model = smf.ols("Salary~YearsExperience",data = salary).fit()
model.summary()

pred=model.params

print(model.tvalues, '\n', model.pvalues)    

(model.rsquared,model.rsquared_adj)

newsalary=pd.Series([30,40])

data_pred=pd.DataFrame(newsalary,columns=['YearsExperience'])
data_pred

model.predict(data_pred)