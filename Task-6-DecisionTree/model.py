import  pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pickle

df=sns.load_dataset('iris')


df.drop_duplicates(inplace=True)

X=df[['sepal_length','sepal_width','petal_length','petal_width']].values
# X[0:5]
Y=df['species']
# Y[0:5]

# Splitting the dataset into train and test data using train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=4)

# modeling the data using Decision Tree
speciesTree=DecisionTreeClassifier(criterion='gini',max_depth=3)

# Fitting the training data into our model
speciesTree.fit(x_train,y_train)

# Accuracy of testing data 
predTree=speciesTree.predict(x_test)
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test,predTree))

file = open("model.pkl","wb")
pickle.dump(speciesTree,file)
file.close()

# input=[6.7,3.0,5.2,2.3]
# print(speciesTree.predict([input])[0])