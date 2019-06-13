
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
%matplotlib inline 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../input/data.csv",header = 0)



df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
from sklearn.neighbors import KNeighborsClassifier
df.diagnosis.unique()
df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})
attribute_mean=list(df.columns[1:11])
df1=df[df['diagnosis'] ==1]
df0=df[df['diagnosis'] ==0]
corr=df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(120, 100, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

x=df['diagnosis']
y=df.loc[:,'radius_mean':'fractal_dimension_mean']
X_train, X_test, y_train, y_test = train_test_split(y, x, stratify=x, random_state=70)
linreg=LinearRegression().fit(X_train, y_train)
print("linear model intercept (b):{}".format(linreg.intercept_))
print("linear model coefficient (b):{}".format(linreg.coef_))
