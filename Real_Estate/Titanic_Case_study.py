
# coding: utf-8

# In[1]:

import pandas as pd


# In[59]:

train_df = pd.read_csv(r"C:\Users\yla880\Downloads\train.csv")


# In[60]:

train_df["child"] = float('NaN')
train_df.head()
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())


# In[61]:

train_df["child"][train_df["Age"]<18]=1


# In[62]:


train_df["child"][train_df["Age"]>=18]=0


# In[63]:

train_df["Survived"][train_df["child"] == 1].value_counts(normalize = True)


# In[64]:

train_df["Survived"][train_df["Sex"] == 'female'].value_counts(normalize = True)


# In[65]:

train_df["Survived"][train_df["Sex"] == 'male'].value_counts(normalize = True)


# In[66]:

train_df.head()


# In[68]:

from sklearn import tree
import numpy as np
# convert sex column to integer
train_df["Sex"][train_df["Sex"] == 'male'] = 0
train_df["Sex"][train_df["Sex"] == 'female'] = 1

train_df["Embarked"] = train_df["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
train_df["Embarked"][train_df["Embarked"] == "S"] = 0
train_df["Embarked"][train_df["Embarked"] == 'C'] = 1
train_df["Embarked"][train_df["Embarked"] == 'Q'] = 2 




# In[117]:

# Create the target
target = train_df["Survived"].values
features_one = train_df[["Pclass", "Sex", "Age", "Fare","Embarked"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
train_df["Survived"][train_df["Embarked"]].value_counts(normalize = True)
train_df["Survived"][train_df["Sex"]].value_counts(normalize = True)
train_df["Survived"][train_df["Age"]].value_counts(normalize = True)



# In[118]:

import numpy as np
my_prediction = my_tree_one.predict(features_one)
print(my_prediction)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(train_df["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv(r"C:\Users\yla880\Documents\Learning\my_solution_one.csv", index_label = ["PassengerId"])


# In[123]:

target = train_df["Survived"]
featured_two = train_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","child"]]
my_tree_two = tree.DecisionTreeClassifier()
my_tree_two = my_tree_two.fit(featured_two,target)

print(my_tree_two.score(featured_two,target))
print(my_tree_two.feature_importances_)
print(my_tree_two.score(featured_two,target))


# In[106]:

pd.value_counts(train_df.values.flatten())
my_prediction_two = my_tree_two.predict(featured_two)
passenger_id = np.array(train_df["PassengerId"]).astype(int)
#print(my_prediction_two)
soln = pd.DataFrame(my_prediction_two,passenger_id,columns = ["Survived"])
train_df.head()
print (soln)
soln.to_csv(r"C:\Users\yla880\Documents\Learning\my_solution_two.csv", index_label = ["PassengerId"])
from sklearn import g


# In[122]:

aa = {}


# In[111]:




# In[112]:




# In[113]:




# In[115]:




# In[ ]:



