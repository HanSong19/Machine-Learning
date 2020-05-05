# Predicting the Qulaity of Wine

This is my first Machine Learning program predicting the quality of red wine.
I have used linear-regression model to predict the quality of wine.

## Description
11 colums for Factors:

1 - fixed acidity

2 - volatile acidity

3 - citric acid

4 - residual sugar

5 - chlorides

6 - free sulfur dioxide

7 - total sulfur dioxide

8 - density

9 - pH

10 - sulphates

11 - alcohol

One output variable:
12 - quality (score between 0 (lowest)-10(highest))

I wanted to make a programme that shows which factor affects the qulity of wines the most.

Hence, I used linear regression model to find the correlation between factor and the quality.

The models I have used are 

1)LinearRegression
2)DecisionTreeRegressor
3)RandomForestRegressor

## Data and Acknowledgement

I have used the data from Kaggle (https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009), and I did NOT create this data set or own it.

Also, the dataset is available from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/wine+quality).
Please include this citation if you plan to use this database: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

**P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


### Installing

Window 10, Python3, Jupyter Notebook, Anaconda

### Executing Programme

Open the file and see how data is formed

```
import pandas as pd
rwine = pd.read_csv("C:/Users/HanByulSong/Desktop/NLP/Kaggle/winequality-red.csv")
rwine.head()
rwine.describe()
```

Visualize each colums using matplotlib

```
%matplotlib inline
import matplotlib.pyplot as plt
rwine.hist(bins = 30, figsize = (20,20))
plt.show()
```

### Prepare Dataset

Split data to Train-set (0.8, 1280 rows) and Test-set (0.2, 319 rows)

```
from sklearn.model_selection import train_test_split

def split_test_train_data(data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))
    test_length= int(len(data)*test_ratio)
    test_indicies= shuffled_indicies[:test_length]
    train_indicies = shuffled_indicies[test_length:]
    return data.iloc[train_indicies], data.iloc[test_indicies]
  
train_set, test_set = split_test_train_data(rwine, test_ratio = 0.2)

```


Visualize the relation among 'alcohol', 'density', and the quality. And check the corrlation

```
rwine.plot(kind = "scatter", x="density", y="alcohol", alpha = 0.1, figsize = (10,7),

            c="quality", cmap=plt.get_cmap("jet"), colorbar=True,
      
)

plt.legend()
corr_matrix["quality"].sort_values(ascending=False)
```

## Running Models


1) Linear model


```

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(rwine, rwine_labels)

from sklearn.metrics import mean_squared_error

rwine_prediction = lin_reg.predict(rwine)
lin_mse = mean_squared_error(rwine_labels, rwine_prediction)
line_rmse = np.sqrt(lin_mse)
line_rmse
```


2) Decision Tree Regressor model


```

from sklearn.tree import DecisionTreeRegressor
[...]
line_rmse

```


2) Random Forest Regressor model


```

from sklearn.ensemble import RandomForestRegressor
[...]
line_rmse

```


## Running the tests

Run cross validation to see the score of each model

```

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, rwine, rwine_labels,
                    scoring = "neg_mean_squared_error", cv=10)
tree_rsme_scores = np.sqrt(-scores)

```

### Fine Tune the model

Random Forest Regresion had the best model and not find tune the model using GridSearchCV

```
from sklearn.model_selection import GridSearchCV

```

### Evalute the model and find 95% confidence rate

Evalute the model using the Test-set.

Then, perform 95% confidence test

```
final_model = freg_search.best_estimator_

[...]
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - rwine_test_labels)**2
```



## Contributing


I have built this model After studying the first chapter of "Hands-on Machine Learning with Scikit-Learn and TensorFlow"
Please read the book for the further information

*citation for the book
Géron, A. (2019). Hands-on machine learning with Scikit-Learn and TensorFlow : concepts, tools, and techniques to build intelligent systems. Sebastopol, CA: O'Reilly Media. ISBN: 978-1491962299



## Acknowledgments

* P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

*P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


