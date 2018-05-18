# project_mc_nulty

The goal of this project is to predict the state of the fridge consumption (on, off) at any precise minute,
given the information about the the total consumption of the house in that minute (aggregate).
The data used for the project are 10 month of active power data sampled every minute from a single building.
The range is from 1 january 2016 to 30 october 2016.
The exploratory data analysis (presented in the EDA jupyter notebook) outlines the choiche of 11 Watt/minute as the threshold for the fridge state.
The classification is done through the use of the Random Forest Algorithm implemented by the scikit-learn library.
This repository contains the following:
- EDA : python notebook of the exploratory data analysis.
- edautils: functions used in the eda to create the plots.
- data_59_all.csv.tar.gz : the data used for this analysis.
- FridgeFeatureExtractor.py : the class responsible for feature extraction.
- FridgeStateClassifier.py : the class responsible of running several models, classifiers an tests
- main.py : example of classifier usage.

The classifier implements five models (numbered from 0 to 4) whose difference is the strategy used to tackle
the class imbalance (9% positive, 91% negative) of the dataset.   
The best results (the one presented in the slides in the doc folder of this repository) are the one in which the data from the positive class (fridge active) are oversampled  through the use of the SMOTE algorithm.
This corresponds to the build_model_4 method of the FridgeStateClassifier class, which gives the following results. 

Metric| Accuracy | Precision| Recall | F1 |
--- | --- | --- | --- |--- |
Value | 0.96 | 0.78 | 0.7 | 0.74 |

