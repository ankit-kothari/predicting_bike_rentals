# predicting_bike_rentals
Predicting Bike Rentals Using Different Machine Learning Algorithms

| Model |	Parameters |	Bagging |	MAE Value
| Linear Regression 	| NA | No | 16802		
| Decision Tree Model 1 |	min_samples_leaf=5	| No |	2737
| Decision Tree Model 1	| min_samples_leaf=2; max_depth=15	| Yes	| 2071
| Random Forest Model 1	| min_samples_leaf=5 max_depth=15	| Yes (Bootstrapping)	| 2248
| Random Forest Model 2	| "min_samples_leaf=2 n_estimators=30" |	Yes ((Bootstrapping) | 1881
