# project_mc_nulty

The goal of this project is to predict the state of the fridge consumption (on, off) at any precise minute,
given the information about the the total consumption of the house in that minute.
The data used for the project are 10 month of data of a single building ranging from 1 january 2016 to 30 october 2016.
This repository contains the following:
- EDA : python notebook of the exploratory data analysis
- edautils: functions used in the eda
- data_59_all.csv.tar.gz : the data used for this analysis
- FridgeFeatureExtractor.py : the class responsible for feature extraction
- FridgeStateClassifier.py : the class responsible of running several models, classifiers an tests
- main.py : example of classifier usage
