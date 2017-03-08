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

X=pd.read_csv('C:\\Users\\juhi\\Documents\\python\\DAT210x-master\\DAT210x-master\\Module6\\Datasets\\parkinsons.data')
X.head()
X=X.drop('name', axis=1)
y=X['status']
X=X.drop('status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
a=np.arange(0.05,2.05,0.05)
b=np.arange(0.001,0.101,0.001)
best_score=0
for i in range(len(a)):
    C=a[i]
    for k in range(len(b)):
        g=b[k]
        svc = SVC(C=C,gamma=g)
        svc.fit(X_train,y_train)
        score=svc.score(X_test,y_test)
        if best_score<score:
            best_score=score
            CF=C
            GF=g
print(best_score,C,g)
        
