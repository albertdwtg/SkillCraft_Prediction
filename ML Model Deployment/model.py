import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
sns.set_theme(style="darkgrid")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import roc_auc_score

df=pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv')
ind = df[ df['LeagueIndex'] ==8].index
df.drop(ind , inplace=True)
ind = df[ df['TotalHours'] =='?'].index
df.drop(ind , inplace=True)
df["Age"]=df["Age"].astype('int')
df["TotalHours"]=df["TotalHours"].astype('int')
df["HoursPerWeek"]=df["HoursPerWeek"].astype('int')

col = df.columns

from sklearn.ensemble import IsolationForest
IsoForest = IsolationForest(contamination=0.03)
IsoForest.fit(df[col])
outliersIso = IsoForest.predict(df[col]) == -1

df['OutlierIso'] = outliersIso
df[df['OutlierIso'] == True]

from sklearn.neighbors import LocalOutlierFactor

LocalOut = LocalOutlierFactor(n_neighbors=4)
outliersLocal = LocalOut.fit_predict(df[col]) == -1

df['OutlierLocal'] = outliersLocal
df[df['OutlierLocal'] == True]

ind = df[df['OutlierLocal'] == True][df['OutlierIso'] == True].index
df.drop(ind , inplace=True)
df = df[col]

df=df.drop([
            "GameID",
            "Age",
            "HoursPerWeek",
            "TotalHours",
            "UniqueHotkeys",
            "MinimapAttacks",
           "MinimapRightClicks",
            "ActionsInPAC",
            "TotalMapExplored",
            "WorkersMade",
            "UniqueUnitsMade",
           "ComplexUnitsMade",
            "ComplexAbilitiesUsed",
           ],axis=1)

df["LeagueIndex"]=df["LeagueIndex"].replace([1,2,3,4,5,6,7],[10,10,10,10,11,11,11])
df["LeagueIndex"]=df["LeagueIndex"].replace([10,11],[0,1])

col = list(df.columns)
col.remove('LeagueIndex')
X = df[col].values
y = df['LeagueIndex'].values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler2=MinMaxScaler()
scaler2.fit(X_train)
X_train2 = scaler2.fit_transform(X_train)
X_test2 = scaler2.transform(X_test)

scaler3=RobustScaler()
scaler3.fit(X_train)
X_train3 = scaler3.fit_transform(X_train)
X_test3 = scaler3.transform(X_test)

scaler4=MaxAbsScaler()
scaler4.fit(X_train)
X_train4 = scaler4.fit_transform(X_train)
X_test4 = scaler4.transform(X_test)

scaler5=Normalizer()
scaler5.fit(X_train)
X_train5 = scaler5.fit_transform(X_train)
X_test5 = scaler5.transform(X_test)

model=GradientBoostingClassifier(max_depth=1, max_features='log2')
model.fit(X_train2,y_train)

import pickle
pickle.dump(model,open("model.pkl","wb"))
print("done")
