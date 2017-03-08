import pandas as pd
from subprocess import call
from sklearn.model_selection import train_test_split
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
# .. your code here ..
X=pd.read_csv('C:\\Users\\juhi\\Documents\\python\\DAT210x-master\\DAT210x-master\\Module6\\Datasets\\agaricus-lepiota.data')
Z=X
# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]


# 
# TODO: Go ahead and drop any row with a nan
#
# .. your code here ..
X=X.dropna()

#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
# .. your code here ..
y=pd.DataFrame(X['p'])
y['p'].unique()
#y.p = X.Private.map({'Yes':1, 'No':0})

a=pd.DataFrame([['p',0],['e',1]])
y.p=y.p.map({'p':1, 'e':0})

X=X.drop('p', axis=1)
X=pd.get_dummies(X)

X.columns
a
#
# TODO: Encode the entire dataset using dummies
#
# .. your code here ..


# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#
# TODO: Create an DT classifier. No need to set any parameters
#
# .. your code here ..
dct = tree.DecisionTreeClassifier()

 
#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
# .. your code here ..
dct.fit(X_train,y_train)
score=dct.score(X_test,y_test)
X=X_test
clf=dct
print ("High-Dimensionality Score: ", round((score*100), 3))
zip(X.columns[clf.tree_.feature], clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right)

print(dct.tree_.children_left) #array of left children
print(dct.tree_.children_right) #array of right children
print(dct.tree_.feature) #array of nodes splitting feature
print(dct.tree_.threshold) #array of nodes splitting points
print(dct.tree_.value) #array of nodes values

dct.tree_
from inspect import getmembers
print( getmembers( dct.tree_ ) )

#
# TODO: Use the code on the courses SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz. If you can't, use: http://webgraphviz.com/
#
# .. your code here ..
tree.export_graphviz(dct.tree_, out_file='tree.dot', feature_names=X.columns)
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

