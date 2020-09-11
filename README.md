# predicting_bike_rentals
Predicting Bike Rentals Using Different Machine Learning Algorithms

## Code

[Google Colaboratory](https://colab.research.google.com/drive/116SH8JJiB4T9fvHb9gCbxirgiOCWmOKd?usp=sharing)

## Performance Criteria

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-08_at_1.31.21_AM.png" width="40%">
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b922fca-b436-4592-af79-ea744bf54e06/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b922fca-b436-4592-af79-ea744bf54e06/Untitled.png)

Formula for Mean Squared Error

## Data Prepration

### Data Loading and Ploting

- The datafram has 17379 rows and 17 columns
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.12.29_AM.png" width="40%">
   

    ```python
    (17379, 17)
    ```

    ```python
    Index(['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
           'casual', 'registered', 'cnt'],
          dtype='object')
    ```

- Count of bikes rented per hour for each of the row in the dataset.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.12.29_AM.png" width="40%">
    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2d8ec8e0-6872-44f5-b396-0507b4238a09/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2d8ec8e0-6872-44f5-b396-0507b4238a09/Untitled.png)

## Feature engineering

### One-Hot Encoding

- Converting the weekday into a Categorical column and then one-hot encoding it into "Monday, Tuesday, Wednesday, Thursday, Saturday, Sunday"
- Converting hr of the day into four categories 'morning', 'aftenoon', 'evening', 'night'
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.43.46_AM.png" width="40%">


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.43.52_AM.png" width="40%">

### Corelations Matrix

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2ba7e76-7e01-4f68-bbb1-3f2494675e97/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2ba7e76-7e01-4f68-bbb1-3f2494675e97/Untitled.png)
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.12.29_AM.png" width="40%">
Corelations between columns 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/736150c5-63b2-4221-a51d-e680c8b2729c/Screen_Shot_2020-07-28_at_12.56.30_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/736150c5-63b2-4221-a51d-e680c8b2729c/Screen_Shot_2020-07-28_at_12.56.30_PM.png)
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_11.12.29_AM.png" width="40%">
Corelations between columns and the target variable ['cnt'] in this case

### Dropping Redundant Columns

- Calculationg correlations for the total bikes rented and removing columns that do not contribute to model performance.
    - dteday is redundant information since we have yr, month, day colums so we can drop it, but it will make sense to use the date column to generate other columns during prediction.
    - Looking at the graph and table below doesn't year doesn't make sense to be a value which should be used to predict for the future
    - 'atemp' and 'temp' columns are highly co-related so we can drop any one,
    - mnth and season convey similar meanings, months is easy to gather from the date value so getting rid of the 'season' column

## Model

### Model 1: Decesion Tree with  (min_samples_leaf=5)

```python
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(min_samples_leaf=5, random_state=1)
dt.fit(train[feature_columns], train["cnt"])

DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=5, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort='deprecated',
                      random_state=1, splitter='best')
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cb7039ba-0716-4a1f-add1-b8a865dd6695/Screen_Shot_2020-07-28_at_3.50.44_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cb7039ba-0716-4a1f-add1-b8a865dd6695/Screen_Shot_2020-07-28_at_3.50.44_PM.png)
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_3.50.44_PM.png">

### Model 2: Decesion Tree with (max_depth=15, min_samples_leaf=5)

- Tree Count = 10
- Each "bag" will have 60% of the number of original rows (bag_proportion = .6)
- We select 60% of the rows from train, sampling with replacement
- We set a random state to ensure we'll be able to replicate our results
- We set it to i instead of a fixed value so we don't get the same sample every time

```python
from sklearn.tree import DecisionTreeRegressor
import numpy
tree_count = 10
# Each "bag" will have 60% of the number of original rows
bag_proportion = .6
predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample every time
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)
    # Fit a decision tree model to the "bag"
    clf = DecisionTreeRegressor(random_state=1, min_samples_leaf=2, max_depth=15)
    clf.fit(bag[feature_columns], bag["cnt"])
    predict = clf.predict(test[feature_columns])
    # Using the model, make predictions on the test data
    predictions.append(predict)

DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=15,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=2, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort='deprecated',
                      random_state=1, splitter='best')
```
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_3.52.34_PM.png">


### Model 3: Random Forest with  (max_depth=15, min_samples_leaf=5)

```python
from sklearn.ensemble import RandomForestRegressor
reg1 = RandomForestRegressor(min_samples_leaf=5,random_state=1, max_depth=15)
reg1.fit(train[feature_columns], train["cnt"])

RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=15, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=5,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=1, verbose=0, warm_start=False)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef0b0575-6abb-4d37-abf8-99090902caaf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef0b0575-6abb-4d37-abf8-99090902caaf/Untitled.png)
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-07-28_at_3.52.34_PM.png">

### Model 4: Random Forest with  (n_estimators=30, min_samples_leaf=2)

```python
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(min_samples_leaf=2,random_state=1, n_estimators=30)
reg.fit(train[feature_columns], train["cnt"])

RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=2,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=30, n_jobs=None, oob_score=False,
                      random_state=1, verbose=0, warm_start=False)
```



## Model Evaluation and Predictions


| Model |	Parameters |	Bagging |	RMSE Value |
| -----| -----------| ----------| ----------|
| Decision Tree Model 1 |	min_samples_leaf=5	| No |	30.39 |
| Decision Tree Model 1	| min_samples_leaf=2; max_depth=15; number of trees: 10	| Yes	| 15.71|
| Random Forest Model 1	| min_samples_leaf=5 max_depth=15	| Yes (Bootstrapping)	| 13.86 |
| Random Forest Model 2	| "min_samples_leaf=2 n_estimators=30" |	Yes ((Bootstrapping) | 11.29 |
