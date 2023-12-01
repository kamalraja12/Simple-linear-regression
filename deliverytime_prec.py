# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:16:55 2023

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

df=pd.read_csv("Downloads\delivery_time.csv")
df

df.describe()

df.corr()

df.hist()

df=df.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)
df

x = df.delivery_time
y = df.sorting_time
plt.scatter(x,y)
plt.xlabel=("delivery_time")
plt.ylabel=("sorting_time")

df.boxplot()
sns.pairplot(df)
sns.distplot(df['delivery_time'])
sns.distplot(df['sorting_time'])
sns.regplot(x='delivery_time', y='sorting_time', data=df)

model=smf.ols("sorting_time~delivery_time ", data=df).fit()
model.summary()
model.params
print(model.tvalues,'\n' ,model.pvalues)
(model.rsquared,model.rsquared_adj)

#Transformation models
model2 = smf.ols("np.log(sorting_time)~delivery_time", data=df).fit() 
model2.params
model2.summary()  
(model2.rsquared,model2.rsquared_adj)

model3 = smf.ols("np.sqrt(sorting_time)~delivery_time", data=df).fit() 
model3.params
model3.summary()
(model3.rsquared,model3.rsquared_adj)

#Prediction
newdata=pd.Series([10,5])
data_pred=pd.DataFrame(newdata, columns=['delivery_time'])
data_pred
model3.predict(data_pred)