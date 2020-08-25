#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_income.csv")
df
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Year")
plt.ylabel("Salary Per Capita")
plt.scatter(df.year,df.income, color = "red", marker = "+")

new_df = df.drop("income",axis="columns")
new_df

model = linear_model.LinearRegression()
model.fit(new_df,df.income)

model.predict([[2020]])





