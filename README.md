# project_mc_nulty

The goal of this project is to predict the state of the fridge consumption (on, off) at any precise minute,
given the information about the the total consumption of the house in that minute (aggregate).

The data used for the project are 10 month of active power data sampled every minute from a single building.

The range is from 1 january 2016 to 30 october 2016.

The exploratory data analysis (presented in the EDA jupyter notebook) outlines the choiche of 11 Watt/minute as the threshold for the fridge state.

In the exploratory data analysis are presented also the following items:

- The distribution of the aggregate .
- The distribution of the air conditioner .
- The distribution of the electric furnace.
- The comparisons of the aggregate with the activation of each of the three appliances over the hours of the day .
- The appliance signatures . 

The classification is done through the use of the Random Forest Algorithm implemented by the scikit-learn library.

This repository contains the following:
- EDA : python notebook of the exploratory data analysis.
- edautils: functions used in the eda to create the plots.
- data_59_all.csv.tar.gz : the data used for this analysis.
- FridgeFeatureExtractor.py : the class responsible for feature extraction, it implements 5 different transformation of the dataset.
- FridgeStateClassifier.py : the class responsible of running several models, classifiers an tests
- main.py : example of classifier usage.

The feature extractor implements five ways to transform the original dataset to tackle the class imbalance (9% positive, 91% negative) of the dataset.   

The analysis presented in the slides is  related to the model number 4, the one in which the data from the positive class (fridge active) are oversampled  through the use of the SMOTE algorithm.

The results are the following:
 
Metric| Accuracy | Precision| Recall | F1 |
--- | --- | --- | --- |--- |
Value | 0.96 | 0.78 | 0.7 | 0.74 |

The area under the ROC curve is 0.98.

The analysis of the models 1, 3 are not presented, as the dataset were filtered with a bias towards the days with activation and, as such, the metric scored higher, but the problem deviated from the original statement.

