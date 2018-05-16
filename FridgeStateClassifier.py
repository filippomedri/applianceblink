import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from FridgeFeatureExtractor import FridgeFeatureExtractor

class FridgeStateClassifier():

    def __init__(self,
                 threshold=12,
                 cv=3,
                 knn_range=20,
                 rf_min_samples_leaf=20,
                 rf_class_weight_0=.3):
        self.threshold = threshold
        self.__cv = cv
        self.__knn_range = knn_range
        self.__rf_min_sample_leaf = rf_min_samples_leaf
        self.__rf_class_weight = {0.0: rf_class_weight_0, 1.0: 1.0-rf_class_weight_0}

        self.__knn_param = None

        self.__rf_parameters = None

        self.classifiers = {}


    def extract(self,data_source):
        self.data_source = data_source
        self.extractor = FridgeFeatureExtractor(self.data_source)

    def transform(self):
        pass

    def __evaluate_knn_param(self):
        k_range = list(range(1, self.__knn_range+1))
        param_grid = dict(n_neighbors=k_range)
        clf = KNeighborsClassifier()
        print("Grid search")
        grid = GridSearchCV(clf, param_grid, cv=self.__cv, scoring='accuracy')
        print("Grid fit")
        print(self.X_train.head(), self.y_train)
        grid.fit(self.X_train, self.y_train)
        # TODO: LOG IT!!!
        print(" KNN best score = ",grid.best_score_)
        print(" KNN best param = ",grid.best_params_)
        print(" KNN best estimator = ",grid.best_estimator_)
        self.knn_param = grid.best_params_.get('n_neighbors',1)

    def __run_knn(self):
        if self.__knn_param is None:
            self.__evaluate_knn_param()

        self.__knnClassifier = KNeighborsClassifier(n_neighbors=self.knn_param)
        self.__knnClassifier.fit(self.X_train, self.y_train)

        self.classifiers['KNN'] = (self.__knnClassifier, 1)

    def __evaluate_rf_param(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 20]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 8, 16]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Class weight
        class_weight = [{0.0: .1, 1.0: .9},{0.0: .2, 1.0: .8},{0.0: .3, 1.0: .7},{0.0: .4, 1.0: .6},{0.0: .5, 1.0: .5}]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'class_weight':class_weight}

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=8,
            cv=self.__cv,
            verbose=2,
            random_state=42,
            n_jobs=-1)
        rf_random.fit(self.X_train,self.y_train)
        print (rf_random.best_params_)
        print(" RF rs best score = ", rf_random.best_score_)
        print(" RF rs best param = ", rf_random.best_params_)
        print(" RF rs best estimator = ", rf_random.best_estimator_)
        self.__rf_parameters = rf_random.best_params_


    def __optimize_rf_parameters(self):
        param_grid = {
            'n_estimators': [800,900,1000,1100,1200],
            'min_samples_split': [4,5,6],
            'min_samples_leaf': [1,2,3],
            'max_depth': [90,110,120]
        }
        rf = RandomForestClassifier()
        grid = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=self.__cv, n_jobs=-1, verbose=2)
        grid.fit(self.X_train, self.y_train)
        print(grid.best_params_)
        print(" RF gs best score = ", grid.best_score_)
        print(" RF gs best param = ", grid.best_params_)
        print(" RF gs best estimator = ", grid.best_estimator_)
        self.__rf_parameters = grid.best_params_
        raise Exception



    def __run_random_forest(self):

        self.__rf_parameters = {'n_estimators': 1000,
                                'min_samples_split': 5,
                                'min_samples_leaf': 2,
                                'max_depth': 110}


        # TODO: Change the parameter evaluation
        if self.__rf_parameters is None:
            self.__evaluate_rf_param()

        else:
            #self.__optimize_rf_parameters()
            pass

        self.__randomForestClassifier = RandomForestClassifier(
            **self.__rf_parameters
        )

        self.__randomForestClassifier.fit(self.X_train,self.y_train)
        print("feature importance =", self.__randomForestClassifier.feature_importances_)

        self.classifiers['RFC'] = (self.__randomForestClassifier, 1)

    def build_model_0(self):
        self.extractor.build_model_0(self.threshold)

    def build_model_4(self):
        self.extractor.build_model_4(self.threshold)

    def build_model_1(self):
        self.extractor.build_model_1(self.threshold)

    def build_model_2(self):
        self.extractor.build_model_2(self.threshold)

    def build_model_3(self,ratio=1):
        self.extractor.build_model_3(self.threshold,ratio)

    def run_classifiers(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.extractor.train_test_split()
        #print("Run knn")
        #self.__run_knn()
        print("Run random forest")
        self.__run_random_forest()


    def evaluate_classifiers(self):
        def get_scores(classifier,pos_label):
            y_pred = classifier.predict(self.X_test)
            scores = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, pos_label=pos_label),
                'Recall': recall_score(self.y_test, y_pred, pos_label=pos_label),
                'F1': f1_score(self.y_test, y_pred, pos_label=pos_label)
            }
            return scores
        scores = [get_scores(classifier, pos_label) for classifier, pos_label in self.classifiers.values()]

        columns = ['Accuracy', 'Precision', 'Recall', 'F1']
        self.score_df = pd.DataFrame(scores, index=self.classifiers.keys(), columns=columns)

        return self.score_df

    def plot_roc_curve_classifiers(self):
        def plot_roc_curve(classifier, classifier_name):
            sns.set(color_codes=True)
            plt.figure(figsize=(11, 5),)
            probabilities = classifier.predict_proba(self.X_test)
            fpr, tpr, thrs = roc_curve(self.y_test, probabilities[:, 1])
            plt.title(classifier_name)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.plot(fpr, tpr)
            plt.show()

        matplotlib.rcParams.update({'font.size': 62})
        for classifier, classifier_name in zip((self.classifiers.values()), self.classifiers.keys()):
            plot_roc_curve(classifier[0], classifier_name)
        plt.savefig('roc_curve.png')

    def get_auc(self):
        def auc(classifier):
            probabilities = classifier.predict_proba(self.X_test)
            return roc_auc_score(self.y_test, probabilities[:, 1])

        auc_scores = [auc(classifier[0]) for classifier in self.classifiers.values()]
        self.auc = pd.DataFrame(auc_scores, index=self.classifiers.keys(), columns=['AUC'])
        print (self.auc)

    def store_models(self):
        pass

    def load(self):
        pass
