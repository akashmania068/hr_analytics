'''
    Predicting whether a selected candidate will join the company or not
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import timeit
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score


class HRAnalytics:

    use_cols = list(range(4, 32))
    use_cols.remove(30)  # The columns to be considered for prediction
    label_classes = 2
    max_train_sample = 10000
    max_test_sample = 10000
    each_class_max_sample = max_train_sample // 2
    model_dict = {}

    def __init__(self, file_path):

        # Initializing the train and test DataFrames
        self.x = pd.DataFrame()
        self.y = pd.Series()
        self.train_dataset = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.y_train = pd.Series()
        self.test_dataset = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.Series()

        self.__read_dataset(file_path)

    def __read_dataset(self, file_path):

        # Reading Dataset
        self.dataset = pd.read_excel(io=file_path, usecols=self.use_cols)
        self.__data_preprocessing()

    def __data_preprocessing(self):

        # Removing Rows containing 'Pipeline' in Target Variable (Given in Problem Statement)
        l = len(self.dataset.index)
        y_dataset = self.dataset.ix[:, 'Joined Status']
        self.dataset = self.dataset.drop([i for i in range(l) if y_dataset[i].lower() == 'pipeline'], axis=0)

        # Renaming some columns (Optional) (Only for my convenience)
        self.dataset = self.dataset.rename(columns={'# Job Changes (in last 4 yrs)': 'Job_Changes_Past_4_Years',
                                                    'Employment Mode': 'Current_Employment_Mode',
                                                    'Size of Company': 'Current_Size_of_Company',
                                                    'Salary Change (CTC Amount in Rs)': 'Current_Salary_Change_(CTC_Amount_in_Rs)',
                                                    ' Offered Salary Change (CTC Amount in Rs)': 'Offered_Salary_Change_(CTC_Amount_in_Rs)',
                                                    'Work Schedule Change': 'Current_Work_Schedule_Change',
                                                    'Location Advantage( Proximity)': 'Current_Location_Advantage_(Proximity)',
                                                    'Offered Location Advantage( Proximity)': 'Offered _Location_Advantage_(Proximity)'})
        self.dataset.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

        # Cleaning and Filling Missing Values of each Feature

        # Interest_Reason
        self.dataset['Interest_Reason'].replace('Brand 0me', 'Brand Name', inplace=True)
        # print(self.dataset['Interest_Reason'].value_counts())
        # print(self.dataset['Interest_Reason'].isnull().sum())

        # Age
        # TODO : Make it categorical
        for i in self.dataset.Age[self.dataset.Age < 18]:
            if i == 0:
                self.dataset['Age'].replace(0, np.average(self.dataset['Age']).round(), inplace=True)
            else:
                self.dataset['Age'].replace(i, np.average(self.dataset.Age[(self.dataset.Age >= i * 10) & (self.dataset.Age < i * 10 + 10)]).round(), inplace=True)
        # print(self.dataset['Age'].value_counts())
        # print(self.dataset['Age'].isnull().sum())

        # Experience_Level
        # TODO : Remove the code below and then compare the result.
        for i in self.dataset['Experience_Level']:
            if i == 0:
                continue
            if (i - math.floor(i)) <= 0.3:
                self.dataset['Experience_Level'].replace(i, math.floor(i), inplace=True)
            elif (i - math.floor(i)) > 0.3 and (i - math.floor(i)) <= 0.5:
                self.dataset['Experience_Level'].replace(i, math.floor(i) + 0.5, inplace=True)
            elif (i - math.floor(i)) > 0.5 and (i - math.floor(i)) <= 0.7:
                self.dataset['Experience_Level'].replace(i, math.floor(i) + 0.5, inplace=True)
            else:
                self.dataset['Experience_Level'].replace(i, math.ceil(i), inplace=True)
        # print(self.dataset['Experience_Level'].value_counts())
        # print(self.dataset['Experience_Level'].isnull().sum())

        # Qualification
        c = 0
        for i, s in enumerate(self.dataset['Qualification']):
            if re.search('[bB].* *[tT][eE][cC][hH]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.Tech', inplace=True)
            elif re.search('[mM].* *[tT][eE][cC][hH]', str(s)) or 'degree' in str(s).lower():
                self.dataset['Qualification'].replace(s, 'M.Tech', inplace=True)
            elif re.search('[bB].* *[eE]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.E', inplace=True)
            elif re.search('[mM].* *[eE]', str(s)):
                self.dataset['Qualification'].replace(s, 'M.E', inplace=True)
            elif re.search('[bB].* *[cC][oO][mM]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.Com', inplace=True)
            elif re.search('[mM].* *[cC][oO]*[mM]', str(s)):
                self.dataset['Qualification'].replace(s, 'M.Com', inplace=True)
            elif re.search('[bB].* *[sS][cC]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.Sc', inplace=True)
            elif re.search('[mM].* *[sS][cC]*', str(s)) or (
                ('master' in str(s).lower()) and ('computer') in str(s).lower()):
                self.dataset['Qualification'].replace(s, 'M.Sc', inplace=True)
            elif re.search('[bB].* *[aA]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.A', inplace=True)
            elif re.search('[mM].* *[aA]', str(s)):
                self.dataset['Qualification'].replace(s, 'M.A', inplace=True)
            elif re.search('[mM].* *[bB].* *[aA]', str(s)) or re.search('[pP].* *[gG].* *[dD].* *[bB]*.* *[mM]',
                                                                        str(s)):
                self.dataset['Qualification'].replace(s, 'MBA', inplace=True)
            elif re.search('[mM].* *[cC].* *[aA]', str(s)):
                self.dataset['Qualification'].replace(s, 'M.CA', inplace=True)
            elif re.search('[bB].* *[bB].* *[mM]', str(s)):
                self.dataset['Qualification'].replace(s, 'B.BM', inplace=True)
            elif re.search('[pP].* *[uU].* *[cC]*', str(s)) or \
                            'under' in str(s).lower() or \
                            'intermediate' in str(s).lower() or \
                            'sslc' in str(s).lower():
                self.dataset['Qualification'].replace(s, 'Under Graduate', inplace=True)
            elif re.search('[dD][iI][pP]*[lL]*[pP]*[oO][mM][aA]', str(s)) or 'ifbi' in str(s).lower():
                self.dataset['Qualification'].replace(s, 'Diploma', inplace=True)
            elif re.search('[gG][rR][aA][dD][uU][dD]*[aA][tT]', str(s)):
                self.dataset['Qualification'].replace(s, 'Graduate', inplace=True)
            elif re.search('[pP][oO][sS][tT].* *[gG][rR][aA][dD][uU][aA][tT]', str(s)) or re.search('[pP].* *[gG]', str(s)):
                self.dataset['Qualification'].replace(s, 'Post Graduate', inplace=True)
            elif str(s) == 'nan':
                c += 1
                if self.dataset.Age[i] < 27:
                    self.dataset['Qualification'].replace(s, 'B.Tech', inplace=True)
                else:
                    self.dataset['Qualification'].replace(s, 'M.Tech', inplace=True)
        # print(self.dataset['Qualification'].value_counts())
        # print(self.dataset['Qualification'].isnull().sum())

        # plt.scatter(self.dataset['Age'], self.dataset['Qualification'])
        # plt.show()

        # Notice_Period_Days
        for i in self.dataset['Notice_Period_Days']:
            self.dataset['Notice_Period_Days'].replace(i, int(5 * round(float(i) / 5)), inplace=True)
            if int(i) >= 1000:
                self.dataset['Notice_Period_Days'].replace(i, int(str(i)[:2]), inplace=True)
        # print(self.dataset['Notice_Period_Days'].value_counts())
        # print(self.dataset['Notice_Period_Days'].isnull().sum())

        # plt.scatter(self.dataset['Notice_Period_Days'], self.dataset['Joined_Status'])
        # plt.show()

        # Contract_Duration
        self.dataset['Contract_Duration'] = self.dataset['Contract_Duration'].round()
        # print(self.dataset['Contract_Duration'].value_counts())
        # print(self.dataset['Contract_Duration'].isnull().sum())

        # Current_Salary_Change_(CTC_Amount_in_Rs)
        for i in self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)']:
            try:
                if int(i) > 0 and int(i) < 100:
                    self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, i*100000, inplace=True)
            except Exception:
                try:
                    self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, re.findall('\d+', i)[0], inplace=True)
                    i = re.findall('\d+', i)[0]
                except Exception:
                    self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, 100000, inplace=True)
                    i = 100000
                if int(i) > 0 and int(i) < 100:
                    self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, i * 100000, inplace=True)
        # print(self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].value_counts())
        # print(self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].isnull().sum())

        # Offered_Salary_Change_(CTC_Amount_in_Rs)
        self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].astype(np.float64)

        for i in self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)']:
            try:
                if int(i) > 0 and int(i) < 100:
                    self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, i*100000, inplace=True)
            except Exception:
                try:
                    self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, re.findall('\d+', i)[0], inplace=True)
                    i = re.findall('\d+', i)[0]
                except Exception:
                    self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, 100000, inplace=True)
                    i = 100000
                if int(i) > 0 and int(i) < 100:
                    self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].replace(i, i * 300000, inplace=True)
        # print(self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].value_counts())
        # print(self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].isnull().sum())

        # Hike
        for i, h in enumerate(self.dataset['Hike']):
            x = self.dataset['Offered_Salary_Change_(CTC_Amount_in_Rs)'].iloc[i]
            y = self.dataset['Current_Salary_Change_(CTC_Amount_in_Rs)'].iloc[i]
            try:
                self.dataset.loc[i, 'Hike'] = float(x) - float(y)
            except Exception:
                print(type(x), type(y), i, h)
        # print(self.dataset['Hike'].value_counts())
        # print(self.dataset['Hike'].isnull().sum())

        # Career_Impact
        self.dataset['Career_Impact'].replace('Brand 0me', 'Brand Name', inplace=True)
        # print(self.dataset['Career_Impact'].value_counts())
        # print(self.dataset['Career_Impact'].isnull().sum())

        self.__categorical_encoding()

        self.__train_test_split(test_size=0.3)

    def __categorical_encoding(self):
        # Encoding all the Categorical Features
        # TODO : Encoding
        # converting data type to 'category'
        # label_encoder = LabelEncoder()
        #
        # for i in self.dataset.columns.values.tolist():
        #     if self.dataset[i].dtype == 'object':
        #         self.dataset[i] = self.dataset[i].astype('category')
        #         self.dataset[i] = label_encoder.fit_transform(self.dataset[i])
        pass

    def __train_test_split(self, test_size=0.3):
        # Splitting Dataset to Train and Test
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.3)

        self.x_train = self.train_dataset.ix[:, :'Joined_Status']
        self.y_train = self.train_dataset.ix[:, 'Joined_Status']
        self.x_test = self.test_dataset.ix[:, :'Joined_Status']
        self.y_test = self.test_dataset.ix[:, 'Joined_Status']

    def model_building(self, model=None):
        # TODO : Model Building
        pass

    def lr_model(self):
        logisticRegression = LogisticRegression()
        logisticRegression.fit(self.x_train, self.y_train)
        y_predicted = logisticRegression.predict(self.x_test)

        acc_score = accuracy_score(self.y_test, y_predicted)
        print("Accuracy Score for LR Model :", acc_score)
        self.model_dict.update({'lr': acc_score})

    def knn_model(self):
        pass

    def kmeans_model(self):
        pass

    def svm_model(self):
        pass

    def naivebayes_model(self):
        pass

    def decision_tree(self):
        pass

    def randomforest_model(self):
        pass

    def kmeans_cluster_model(self):
        pass

    def kminibatch_cluster_model(self):
        pass

    def hierarchical_cluster(self):
        pass

    def pca_model(self):
        pass

    def lda_model(self):
        pass

    def feed_forward_nn_model(self):
        pass

    def convolution_nn_model(self):
        pass


hr_analytics = HRAnalytics('Dataset\staffing.xlsx')
hr_analytics.model_building()
