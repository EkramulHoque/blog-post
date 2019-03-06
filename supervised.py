#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


from  sklearn import  datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target


# In[3]:


#labels for iris dataset
labels ={
  0: "setosa",
  1: "versicolor",
  2: "virginica"
}


# In[4]:


#split the data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)


# In[7]:


#Using decision tree algorithm
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)


# In[8]:


#exporting our model
pickle.dump(classifier, open('model.pkl','wb'))


# In[17]:


#load our model and test with a custom input
model = pickle.load( open('model.pkl','rb'))
x = [[6.7, 3.3, 5.7, 2.1]]
print(x_test[70])
print(y_test[70])
predict = model.predict(x)
print(type(predict[0].tolist()))

