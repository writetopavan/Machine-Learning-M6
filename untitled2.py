# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 23:43:49 2016

@author: juhi
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold
X=pd.read_csv('C:\\Users\\juhi\\Documents\\python\\DAT210x-master\\DAT210x-master\\Module6\\Datasets\\parkinsons.data')
X.head()
X=X.drop('name', axis=1)
y=X['status']
X=X.drop('status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

scaler=preprocessing.StandardScaler() # 0.93220338983050843
#scaler=preprocessing.Normalizer() 0.79661016949152541   ,, KernelCenterer()
#scaler=preprocessing.MaxAbsScaler() 0.86440677966101698
#scaler=preprocessing.MinMaxScaler() 0.88135593220338981
#scaler=preprocessing.KernelCenterer()
scaler.fit(X_train)
scaled = scaler.transform(X_train)
scaled1 = scaler.transform(X_test)
X_train = pd.DataFrame(scaled, columns=X_train.columns)
X_test = pd.DataFrame(scaled1, columns=X_test.columns)
'''
model = PCA(n_components=14) # 4 0.88135593220338981  14  0.93220338983050843
model.fit(X_train)
PCA(copy=True, n_components=14, whiten=False)
#2,4 0.9152542372881356, 4,4 0.93220338983050843
'''

model=manifold.Isomap(n_neighbors=5, n_components=6) 
model.fit(X_train)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=2, n_neighbors=4,
    neighbors_algorithm='auto', path_method='auto', tol=0)
    
T = model.transform(X_train)
X_train=pd.DataFrame(T)
T1 = model.transform(X_test)
X_test=pd.DataFrame(T1)

svc = SVC(C=2.0,gamma=0.1)
svc.fit(X_train,y_train)
score=svc.score(X_test,y_test)
score   
        
