import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import imblearn.over_sampling
import imblearn.under_sampling

class FridgeFeatureExtractor():
    def __init__(self, csv_filename):
        '''
        Read dataset from csv 'csv_filename'
        3 cols [Datetime, Aggregate, Fridge]
        Convert Datetime field to datetime type
        '''
        self.fridge_df = pd.read_csv(csv_filename)
        self.fridge_df['Datetime'] = pd.to_datetime(self.fridge_df['Datetime'])

    def train_test_split(self):
        '''
        Return train test split
        :return: X_train, X_test, y_train, y_test
        '''
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model_0(self,threshold):
        '''
        Simple feature model creation
        [Aggregate, Month, Day, Hour, Minute] for feature set, Fridge for target
        Fridge values are binaries values [0,1] corresponding to values below and over the threshold respectively
        :param threshold: watt X minute value that distinguish fridge activation from background noise
        '''

        assert threshold > 0

        self.threshold = threshold
        self.fridge_df.loc[self.fridge_df['Fridge'] < threshold, 'Fridge'] = 0
        self.fridge_df.loc[self.fridge_df['Fridge'] >= threshold, 'Fridge'] = 1

        self.fridge_df['Month'] = self.fridge_df['Datetime'].dt.month
        self.fridge_df['Day'] = self.fridge_df['Datetime'].dt.weekday
        self.fridge_df['Hour'] = self.fridge_df['Datetime'].dt.hour
        self.fridge_df['Minute'] = self.fridge_df['Datetime'].dt.minute

        self.features = self.fridge_df[['Aggregate', 'Month', 'Day', 'Hour', 'Minute']].copy()
        target = self.fridge_df['Fridge'].copy()
        self.target = target.ravel()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=.25,
            random_state=42)

    def build_model_4(self,threshold):
        '''
        Creation of model with SMOTE oversampling
        [Aggregate, Month, Day, Hour, Minute] for feature set, Fridge for target
        Fridge values are binaries values [0,1] corresponding to values below and over the threshold respectively
        :param threshold: watt X minute value that distinguish fridge activation from background noise
        '''
        self.build_model_0(threshold)

        """
        RUS = imblearn.under_sampling.RandomUnderSampler(ratio=.6, random_state=42)
        self.X_train, self.y_train = RUS.fit_sample(self.X_train, self.y_train)
        """
        smote = imblearn.over_sampling.SMOTE(ratio=.35, random_state=42)
        self.X_train, self.y_train = smote.fit_sample(self.X_train, self.y_train)



    def __preprocess(self,threshold):
        '''
        Function shared between several model builder.
        It select record related only to days in which there have been at least an activation
        '''
        self.threshold = threshold
        self.fridge_df.loc[self.fridge_df['Fridge'] < threshold, 'Fridge'] = 0
        self.fridge_df.loc[self.fridge_df['Fridge'] >= threshold, 'Fridge'] = 1

        self.fridge_df['Month'] = self.fridge_df['Datetime'].dt.month
        self.fridge_df['Day'] = self.fridge_df['Datetime'].dt.weekday
        self.fridge_df['Hour'] = self.fridge_df['Datetime'].dt.hour
        self.fridge_df['Minute'] = self.fridge_df['Datetime'].dt.minute

        fridge_daily = self.fridge_df.set_index('Datetime').groupby(pd.Grouper(freq='D'))['Fridge'].max().reset_index()
        fridge_daily_active = fridge_daily[fridge_daily.Fridge > 0]
        fridge_daily_not_active = fridge_daily[fridge_daily.Fridge < 1]
        active_days = fridge_daily_active['Datetime'].dt.date.tolist()
        not_active_days = fridge_daily_not_active['Datetime'].dt.date.tolist()
        self.fridge_df_active = self.fridge_df[self.fridge_df['Datetime'].dt.date.isin(active_days)].copy()
        self.fridge_df_not_active = self.fridge_df[self.fridge_df['Datetime'].dt.date.isin(not_active_days)].copy()


    def build_model_1(self,threshold):
        '''
        Model creation , records are from days with at least one activation
        [Aggregate, Month, Day, Hour, Minute] for feature set, Fridge for target
        Fridge values are binaries values [0,1] corresponding to values below and over the threshold respectively
        :param threshold: watt X minute value that distinguish fridge activation from background noise
         '''
        # Create features = ['Aggregate','Month','Day','Hour','Minute']
        # Create target = ['Fridge'], binary with class 0 items < threshold, class 0 items >= threshold
        self.__preprocess(threshold)

        self.features = self.fridge_df_active[['Aggregate', 'Month', 'Day', 'Hour', 'Minute']].copy()
        target = self.fridge_df_active['Fridge'].copy()
        self.target = target.ravel()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=.25,
            random_state=42)

    def build_model_2(self, threshold):
        '''
                Model creation , records are from days with at least one activation and with days without
                activation in equal part.
                [Aggregate, Month, Day, Hour, Minute] for feature set, Fridge for target
                Fridge values are binaries values [0,1] corresponding to values below and over the threshold respectively
                :param threshold: watt X minute value that distinguish fridge activation from background noise
        '''

        # Create features = ['Aggregate','Month','Day','Hour','Minute']
        # Create target = ['Fridge'], binary with class 0 items < threshold, class 0 items >= threshold
        self.__preprocess(threshold)

        print("Total number of active records = ", len(self.fridge_df_active))

        act_not_act_df = pd.concat(
            [
                self.fridge_df_active,
                self.fridge_df_not_active.sample(len(self.fridge_df_active))
            ]
        )

        print("Total number of records = ", len(act_not_act_df))

        self.features =  act_not_act_df[['Aggregate', 'Month', 'Day', 'Hour', 'Minute']].copy()
        target =  act_not_act_df['Fridge'].copy()
        self.target = target.ravel()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=.25,
            random_state=42)


    def build_model_3(self, threshold,ratio):
        '''
        Model creation , records are from days with at least one activation and with days without
        activation. The ratio express how many records to choose from the days without activation
        eith respect to the days with activation(e.g.: ratio = 3 means set of days without activation is 3 times larger)
        [Aggregate, Month, Day, Hour, Minute] for feature set, Fridge for target
        Fridge values are binaries values [0,1] corresponding to values below and over the threshold respectively
        :param threshold: watt X minute value that distinguish fridge activation from background noise
        '''
        # Create features = ['Aggregate','Month','Day','Hour','Minute']
        # Create target = ['Fridge'], binary with class 0 items < threshold, class 0 items >= threshold

        self.threshold = threshold
        self.fridge_df.loc[self.fridge_df['Fridge'] < threshold, 'Fridge'] = 0
        self.fridge_df.loc[self.fridge_df['Fridge'] >= threshold, 'Fridge'] = 1

        self.fridge_df['Month'] = self.fridge_df['Datetime'].dt.month
        self.fridge_df['Day'] = self.fridge_df['Datetime'].dt.weekday
        self.fridge_df['Hour'] = self.fridge_df['Datetime'].dt.hour
        self.fridge_df['Minute'] = self.fridge_df['Datetime'].dt.minute

        self.fridge_df_active = self.fridge_df[self.fridge_df.Fridge > 0]
        self.fridge_df_not_active = self.fridge_df[self.fridge_df.Fridge < 1]

        #active_days = fridge_minute_active['Datetime'].dt.date.tolist()
        #not_active_days = fridge_minute_not_active['Datetime'].dt.date.tolist()
        #self.fridge_df_active = self.fridge_df[self.fridge_df['Datetime'].dt.date.isin(active_days)].copy()
        #self.fridge_df_not_active = self.fridge_df[self.fridge_df['Datetime'].dt.date.isin(not_active_days)].copy()

        print("Total number of active records = ", len(self.fridge_df_active))
        act_not_act_df = pd.concat(
            [
                self.fridge_df_active,
                self.fridge_df_not_active.sample(ratio*len(self.fridge_df_active))
            ]
        )

        print("Total number of records = ", len(act_not_act_df))

        self.features =  act_not_act_df[['Aggregate', 'Month', 'Day', 'Hour', 'Minute']].copy()
        target =  act_not_act_df['Fridge'].copy()
        self.target = target.ravel()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=.25,
            random_state=42)