#!/usr/bin/env python
# coding: utf-8

# In[1]:


# bring the data 
import pandas as pd


rwine = pd.read_csv("C:/Users/HanByulSong/Desktop/NLP/Kaggle/winequality-red.csv")
rwine.head()


# In[2]:


rwine.info()


# In[3]:


rwine["quality"].value_counts()


# In[4]:


rwine.describe()


# In[5]:


rwine.describe


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
rwine.hist(bins = 30, figsize = (20,20))
plt.show()


# In[7]:


import numpy as np

def split_test_train_data(data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))
    test_length= int(len(data)*test_ratio)
    test_indicies= shuffled_indicies[:test_length]
    train_indicies = shuffled_indicies[test_length:]
    return data.iloc[train_indicies], data.iloc[test_indicies]
    
    
    
    
  
    


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


train_set, test_set = split_test_train_data(rwine, test_ratio = 0.2)
len(train_set)


# In[10]:


len(test_set)
# But the problem here is that if I run the program again, it will give different set of train and test sets and eventually 
# my algorithm will see the entire dataset, and that can cause overfittiing
# so there are are few things I can do to avoid this


# In[69]:


#Making a test set and the training set depending on the specific feature on the data#
#Here, I think alchol is the import factor in findin the price of alcohol
#but alcholi is the numerical data, so I have changed it to categorial and the histogram above showed that the
# alcohol level is most located around 9-10 so bins are from 8-12 



rwine["alcohol_cat"] = pd.cut(rwine["alcohol"],
                              bins = [8.0, 9.0, 10.0, 11.0, 12.0, np.inf],
                              labels = [1,2,3,4,5])





rwine["alcohol_cat"].hist()

print(list(rwine))


# In[12]:


len(rwine.index)


# In[13]:


print(train_set)


# In[14]:


len(train_set)


# In[15]:


len(train_set.index)


# In[16]:


#now I want to make sure that each alcohol category has enough dataset so the data is not scewed

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(rwine, rwine["alcohol_cat"]):
    strat_train_set = rwine.loc[train_index]
    strat_test_set= rwine.loc[test_index]
    
strat_test_set["alcohol_cat"].value_counts() / len(strat_test_set)

## To see the result, if I want to return to the previous non-specific test set, then I can use following code.

for set_ in (strat_train_set, strat_test_set):
    set_.drop("alcohol_cat", axis = 1, inplace = True)


# In[17]:


len(test_index)


# In[18]:


## visualise the data to understand the relation among alcohol, density, and the quality

rwine.plot(kind = "scatter", x="density", y="alcohol", alpha = 0.1, figsize = (10,7),

            c="quality", cmap=plt.get_cmap("jet"), colorbar=True,
      
)

plt.legend()



# In[19]:


#looking at correlations: to see which one affect the medium of rht house value the most
corr_matrix = rwine.corr()


# In[20]:


corr_matrix["quality"].sort_values(ascending=False)


# In[21]:


### Thesea are two test data set rwine = without quality
                                #rwine_labels = only quality
rwine = strat_train_set.drop("quality", axis = 1)
rwine_labels = strat_train_set["quality"].copy()




# In[25]:


######### select and train the model#########
##model 1. Regression model##

#call linear regression model and train
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(rwine, rwine_labels)


# In[26]:


#try it in sample training set#
#purpose of this step is to check whether the linear model works before applying it to the full traing set
some_data = rwine.iloc[:5]
some_labels = rwine_labels.iloc[:5]


print("Rredictions:", lin_reg.predict(some_data))
print("Labels:" , list(some_labels))


# In[27]:


# measure this regression models RMSE#
from sklearn.metrics import mean_squared_error

rwine_prediction = lin_reg.predict(rwine)
lin_mse = mean_squared_error(rwine_labels, rwine_prediction)
line_rmse = np.sqrt(lin_mse)
line_rmse


# In[28]:


#RMSE was better ecause mediam_housing_values ranged 8-12
# and RMSE predicted 0.64, but not satisfying, it is underfitting
# to make a better prediction, this time, use DecisionTreeRegression (a bit more powerful model)


# In[29]:


###model 2. DecisionTreeRegression ##
#Call DecisionTreeRegression and train model


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(rwine, rwine_labels)


# In[30]:


#try DecisionTreeRegressor to sample training set

rwine_prediction = tree_reg.predict(rwine)
tree_mse = mean_squared_error(rwine_labels, rwine_prediction)
tree_rsme = np.sqrt(tree_mse)
tree_rsme


#it shows that the error is 0, whih is not right because it is very unlikely that the error is 0
# it is likely that the moedl overfitted to the training set, so we use part of training set to 
# test it to see if the DecisionTreeRegression overfitted or not 
#so we try Cross-Validation


# In[31]:


# The value of Decision Tree Regressor is 0.0, does this mean that the model is perfect and no error?
# or does it mean that it is over fitting? to check this, I need to compare the score using cross_val, rmse 
# in order to find out, I need to evaluate the score

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, rwine, rwine_labels,
                    scoring = "neg_mean_squared_error", cv=10)
tree_rsme_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:" ,scores)
    print("mean:", scores.mean())
    print("Standard Deviation", scores.std())
    
Tree_reg_score=display_scores(tree_rsme_scores)


# In[32]:


#turnd out that error is even bigger than RMSE model (0.6 vs. here 0.8)
#which is even larger mean than Linear Regression model 

#Then find the score of Linear Regression model

lin_scores = cross_val_score(lin_reg, rwine, rwine_labels,
                             scoring = "neg_mean_squared_error", cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
lin_reg_score = display_scores(lin_rmse_scores)
lin_reg_score


# In[33]:


# so it seems that the score of linear regression is better than Decision Tree Regression
# hence, Decision Tree Regrssion was overfitted


# In[50]:


## model3. RandomForestRegression##
## Building Ensemble Learning: Use Random Forest Regression by training many Decision Trees on random subsets
## See their error/ mean/ sd and score and check with Lin Reg

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(rwine, rwine_labels)

forest_prediction = forest_reg.predict(rwine)
forest_mse = mean_squared_error(rwine_labels, forest_prediction)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

forest_score = cross_val_score(forest_reg, rwine, rwine_labels,
                              scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_score = np.sqrt(-forest_score)
forest_reg_score = display_scores(forest_rmse_score)
forest_reg_score

## seems better as the error is only 0.2 and the score is 0.61


# In[51]:


#save each model#
#lets say there is a folder "my_model"
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")


# In[52]:


my_model_loaded = joblib.load("my_model.pkl")


# In[ ]:


## After I choose the model, in my case either linear or random foreset had better score
## then I need to find-tune the model
#GridSearchCV in Scikit-Learne will evaluate all the possible combination of hyperparameters


# In[53]:


########Fine-Tune my model########
####Grid Search- Random Forest
##Grid Search is when the combination is small
# Here, 3x4 and 2x3, so faily small combination

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators" : [3,10,30], "max_features" : [2,4,6,8]},
    {"bootstrap" :[False], "n_estimators" : [3,10], "max_features" : [2,3,4]},
]

freg_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                          scoring = "neg_mean_squared_error",
                          return_train_score = True)
freg_search.fit(rwine, rwine_labels)


# In[54]:


# Now, I have fit the model, then find the best solution by setting max_feature  and n_estimators

freg_search.best_params_

#This means that the maximum values are 2 and 30
#then try again to find the higher values using estimator


# In[55]:


freg_search.best_estimator_


# In[56]:


# Find the RMSE score for this combination

cvres=freg_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


# since it said that the max is 2 and 30, the value in max_feature:2, and n_estimators:30 
# shows slightly better result (0.600) than (forest_reg: 0.617, lin_reg:0.646 )


# In[73]:


## 2. Fine Tuning: Ensemble Method
## Analyze the best models and errors
# display the importance scores next to their corresponding attributes:

feature_importances = freg_search.best_estimator_.feature_importances_
feature_importances


# In[77]:



num_attribs = list(rwine)
num_attribs
#from this result, I can see and drop less important factors


# In[78]:


sorted(zip(feature_importances, num_attribs), reverse = True)

#from this result, I can see and drop less important factors


# In[79]:


######Evaluate my system on the Test Set ###########
#After model has the lowest error, it is time to evaluate the final model on the test set.

final_model = freg_search.best_estimator_

rwine_test = strat_test_set.drop("quality", axis =1)
rwine_test_labels = strat_test_set["quality"].copy()

final_predictions = final_model.predict(rwine_test)
final_mse = mean_squared_error(rwine_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)



final_rmse


# In[81]:


# to conduct 95% confidence interval test

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - rwine_test_labels)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors) -1,
                        loc=squared_errors.mean(),
                        scale = stats.sem(squared_errors)))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




