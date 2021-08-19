#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("C:\\Users\\HP\\Desktop\\titanic.csv")


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


del_cols = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']


# In[8]:


df  = df.drop(del_cols,1)


# In[9]:


df.columns


# In[10]:


df = df.fillna(df["Age"].mean())


# In[11]:


df.info()


# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[13]:


df["Sex"] = le.fit_transform(df["Sex"])


# In[14]:


df.head()


# In[15]:


df.shape


# In[16]:


df.columns = ['output', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']


# In[17]:


df.head()


# In[18]:


def entropy(col):
    counts = np.unique(col,return_counts=True)
    ent = 0.0
    for ix in counts[1]:
        p = ix/col.shape[0]
        ent += (-1.0*p*np.log2(p))
    return ent


# In[19]:


def divide_data(x_data,fkey,fval):
    x_right = pd.DataFrame([],columns=x_data.columns)
    x_left = pd.DataFrame([],columns=x_data.columns)
    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]
        if val >=fval:
            x_right = x_right.append(x_data.iloc[ix])
        else:
            x_left = x_left.append(x_data.iloc[ix])
    return x_right,x_left


# In[20]:


def information_gain(x_data,fkey,fval):
    right,left = divide_data(x_data,fkey,fval)
    
    l = float(left.shape[0])/x_data.shape[0]
    r = float(right.shape[0])/x_data.shape[0]
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -99999
    i_gain = entropy(x_data.output) - (l * entropy(left.output) + r*entropy(right.output))
    return i_gain


# In[21]:


class DecisionTree:
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.depth = depth
        self.max_depth = max_depth
        self.target = None
    def train(self,x_train):
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
        info_gains = []
        for ix in features:
            i_gain = information_gain(x_train,ix,x_train[ix].mean())
            info_gains.append(i_gain)
        self.fkey = features[np.argmax(info_gains)]
        self.fval = x_train[self.fkey].mean()
        print("Splitting Tree",self.fkey)
        data_right,data_left = divide_data(x_train,self.fkey,self.fval)
        data_right = data_right.reset_index(drop=True)
        data_left = data_left.reset_index(drop=True)
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if x_train.output.mean() >= 0.5:
                self.target = "Positive"
            else:
                self.target = "Negative"
            return
        if self.depth >= self.max_depth:
            if x_train.output.mean() >= 0.5:
                self.target = "Positive"
            else:
                self.target = "Negative"
            return
        self.left = DecisionTree(self.depth+1,self.max_depth)
        self.left.train(data_left)
        self.right = DecisionTree(self.depth+1,self.max_depth)
        self.right.train(data_right)
        if x_train.output.mean() >= 0.5:
            self.target = "Positive"
        else:
            self.target = "Negative"
        return
    def predict(self,test):
        if test[self.fkey] > self.fval:
            if self.right is None:
                return self.target
            return self.right.predict(test)
        if test[self.fkey] < self.fval:
            if self.left is None:
                return self.target
            return self.left.predict(test)


# In[22]:


split = int(0.7*df.shape[0])
train_data = df[:split]
test_data = df[split:]
test_data= test_data.reset_index(drop=True)


# In[23]:


dt = DecisionTree()


# In[24]:


dt.train(train_data)


# In[25]:


y_pred = []
for ix in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[ix]))


# In[26]:


y_pred[:10]


# In[27]:


for i in range(len(y_pred)):
    if y_pred[i] == "Negative":
        y_pred[i] = 0
    else:
        y_pred[i] = 1


# In[28]:


np.mean(y_pred == test_data['output'])


# In[ ]:




