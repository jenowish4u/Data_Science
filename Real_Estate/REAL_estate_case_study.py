
# coding: utf-8

# In[34]:

import pandas as pd
from sklearn import tree
from sklearn import linear_model
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt


# In[66]:

data = pd.read_csv(r"C:\Users\yla880\Documents\Learning\Real_Estate_CaseStudy\train.csv")


# In[67]:

# Target which is going to predict by the system
target = data["SalePrice"]


# In[68]:

data.head()


# In[86]:

#Connect with Categorical and its dependecy with Saleprice(Target)
Work_data = data
data["SaleCondition"][data["SaleCondition"] == 'Partial'].value_counts()
Work_data["SaleCondition_nw"] = 0 
#get distinct values in a particular Column
set(Work_data["SaleCondition"])
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'Abnorml'] = 0
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'AdjLand'] = 1
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'Alloca'] = 2
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'Family'] = 3
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'Normal'] = 4
Work_data["SaleCondition_nw"][Work_data["SaleCondition"] == 'Partial'] = 5

#convert street type 1-Pave 0-Grvl --- No change
Work_data["Street_nw"] = 0
Work_data["Street_nw"][Work_data["Street"] == 'Pave'] =1
#LOT aREA VALUE 

Work_data["LotArea"] = Work_data["LotArea"].fillna(Work_data["LotArea"].median())
Work_data["LotFrontage"] = Work_data["LotFrontage"].fillna(Work_data["LotFrontage"].median())
Work_data["Fireplaces"] = Work_data["Fireplaces"].fillna(0)
Work_data["GarageCars"] = Work_data["GarageCars"].fillna(0)

#lot Shape -- no change
Work_data["LotShape_nw"] = 0
Work_data["LotShape_nw"][Work_data["LotShape"] == 'IR1'] = 0
Work_data["LotShape_nw"][Work_data["LotShape"] == 'IR2'] = 1
Work_data["LotShape_nw"][Work_data["LotShape"] == 'IR3'] = 2
Work_data["LotShape_nw"][Work_data["LotShape"] == 'Reg'] = 3


# In[91]:

dt = linear_model.LinearRegression()
featured_values = Work_data[["YrSold","SaleCondition_nw","LotArea","LandContour_nw","LotFrontage","Fireplaces","GarageCars","BedroomAbvGr","KitchenAbvGr","FullBath","HalfBath","2ndFlrSF","MSSubClass","TotalBsmtSF","LowQualFinSF","GrLivArea","1stFlrSF","OverallCond","YearBuilt","YearRemodAdd","OverallQual"]].values
dt = dt.fit(featured_values,target)
print (dt.score(featured_values,target))


# In[90]:

#Work_data["Street_loc"][Work_data["Street"] == 'Pave'] = 1
#Work_data["Street_loc"][Work_data["Street"] == 'Grvl'] = 0
set(Work_data["LandContour"])

Work_data["LandContour_nw"] = 0
Work_data["LandContour_nw"][Work_data["LandContour"] == 'Bnk'] = 0
Work_data["LandContour_nw"][Work_data["LandContour"] == 'HLS'] = 1
Work_data["LandContour_nw"][Work_data["LandContour"] == 'Low'] = 2
Work_data["LandContour_nw"][Work_data["LandContour"] == 'Lvl'] = 3


# In[ ]:



