# -- coding:utf-8 --
import pandas as pd
import numpy as np
# from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

import time

simsun = FontProperties(fname=r"..\simsun.ttc", size=12)
# plt.rcParams['font.family']='simsun'
# plt.rcParams['axes.unicode_minus']=False # 处理负号问题
# plt.rcParams['font.sans-serif']=['simsun']
pd.options.display.max_columns = None

print(matplotlib.matplotlib_fname())


class Explore:
    def __init__(self, csv_path, col_id, col_result, type_cols=None, serial_cols=None):
        # csv数据路径
        if serial_cols is None:
            serial_cols = []
        if type_cols is None:
            type_cols = []
        self.csv_path = csv_path
        # 数据id列名
        self.col_id = col_id
        # 数据分类结果列
        self.col_result = col_result
        # 类型相关列
        self.type_cols = type_cols
        # 连续数字列
        self.serial_cols = serial_cols
        # 读取csv数据
        print("csv data analysis:")
        self.csv_data = pd.read_csv(self.csv_path)
        print(self.csv_data.head(5))
        print("csv_data info:")
        self.csv_data.info()
        # 分类种类
        if self.col_result is not None:
            self.result_types = self.csv_data.groupby(by=self.col_result).groups.keys()
        # 获取columns
        self.columns = self.csv_data.columns.values.tolist()

    def scan_type(self, type_cols=None):
        if type_cols is not None:
            self.type_cols = type_cols
        # 分析各类型与结果的数据
        fig = plt.figure()
        fig.set(alpha=0.2)
        print("analysis type columns:" + ",".join(self.type_cols))
        if len(self.type_cols) > 0:
            plt.title(u'Type columns analysis images')
            plt_location = 0
            for column in self.type_cols:
                type_counts_dict = {}
                for type_data in self.result_types:
                    type_counts_dict[type_data] = self.csv_data.loc[self.csv_data[self.col_result] == type_data][
                        column].value_counts()
                # 设置绘制图像的信息
                # print("type_counts_dict:")
                # print(type_counts_dict)
                ax = plt.subplot2grid((len(self.type_cols), 1), (plt_location, 0))
                pd.DataFrame(type_counts_dict).plot(kind='bar', ax=ax,
                                                    figsize=(5, 5 * len(self.type_cols)))
                plt.legend(type_counts_dict, prop=simsun)
                plt.title(self.col_result + " by " + column, fontproperties=simsun)
                plt.xlabel(column, fontproperties=simsun)
                plt.ylabel("numbers", fontproperties=simsun)
                plt_location += 1
            plt.show()


    def scan_type_split(self, type_cols=None):
        if type_cols is not None:
            self.type_cols = type_cols
        # 分析各类型与结果的数据
        fig = plt.figure()
        fig.set(alpha=0.2)
        plt.figure(figsize=(5, 5 * len(self.type_cols)))
        print("analysis type columns:" + ",".join(self.type_cols))
        if len(self.type_cols) > 0:
            plt.title(u'Type columns analysis images')
            plt_location = 0
            for column in self.type_cols:
                type_counts_dict = {}
                for type_data in self.result_types:
                    type_counts_dict[type_data] = self.csv_data.loc[self.csv_data[self.col_result] == type_data][
                        column].value_counts()
                # 设置绘制图像的信息
                print("type_counts_dict:")
                print(type_counts_dict)
                c = pd.concat(type_counts_dict, join='outer', axis=1, sort=True)
                c.columns = self.result_types
                c = c.fillna(0)
                print("======" + str(column) + "=====")
                print(c)
                width = 0.8 / len(type_counts_dict)
                # fig, ax = plt.subplots(figsize=(10, 5))
                ax = plt.subplot2grid((len(self.type_cols), 1), (plt_location, 0))
                x = np.arange(len(c))
                for idx, type_data in enumerate(self.result_types):
                    print(type_data)
                    rect = ax.bar(x - width * (len(self.result_types) / 2 - 1 / 2 - idx),
                                  pd.Series(c.to_dict()[type_data]), width, label=type_data)
                    self.autolabel(rect, ax)
                ax.set_ylabel(column)
                ax.set_title(column + ' by ' + self.col_result)
                ax.set_xticks(x)
                plt_location += 1
            plt.show()

    def scan_serial(self, serial_cols=None):
        if serial_cols is not None:
            self.serial_cols = serial_cols
        fig = plt.figure()
        fig.set(alpha=0.2)
        if len(self.serial_cols) > 0:
            plt.title(u'Serial columns analysis images')
            for column in self.serial_cols:
                image_data = {}
                for type_data in self.result_types:
                    image_data[type_data] = self.csv_data.loc[self.csv_data[self.col_result] == type_data][column].plot(
                        kind='kde')
                # 设置绘制图像的信息
                plt.xlabel(column)  # plots an axis lable
                plt.ylabel(u"Distribution")
                plt.title(self.col_result)
                plt.legend(self.result_types, loc='best')
                print(image_data)
            plt.show()

    def scan_heatmap(self, drop_cols=[]):
        drop_cols.append(self.col_id)
        fig_size = len(self.columns) - len(drop_cols)
        fig, axes = plt.subplots(1, 1, figsize=(fig_size*0.7, fig_size*0.7))
        # ax = axes.flatten()
        sns.heatmap(self.csv_data.drop(drop_cols, axis=1).corr(), vmax=0.6,
                    square=True, annot=True, ax=axes)
        plt.show()
        plt.figure()
        plot_list = self.columns.copy()
        for drop_col in drop_cols:
            plot_list.remove(drop_col)
        sns.set(style='ticks', color_codes=True)
        g = sns.pairplot(data=self.csv_data.dropna(), vars=plot_list,
                         hue=self.col_result, size=1.5)
        g.set(xticklabels=[])
        plt.show()


    def autolabel(self, rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def data_preprocess(self):
        self.set_missing_ages()
        self.set_cabin_type()
        self.convert_data(['Cabin', 'Embarked', 'Sex', 'Pclass'])
        print(self.csv_data.head(5))
        self.standard_data(['Age', 'Fare'])
        print(self.csv_data.head(5))
        self.csv_data.drop(['Name', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)


    def standard_data(self, stand_cols):
        scaler = preprocessing.StandardScaler()
        for col in stand_cols:
            col_data = np.array(self.csv_data[col])
            col_data[np.isnan(col_data)] = np.nanmean(col_data)
            scale_col = scaler.fit(col_data.reshape(1, -1))
            result = scaler.fit_transform(col_data.reshape(-1, 1), scale_col)
            result[np.isnan(result)] = np.nanmedian(result)
            self.csv_data[col + "_scaled"] = np.array(result).reshape(-1, 1)


    def convert_data(self, conv_cols):
        for col in conv_cols:
            dump_col = pd.get_dummies(self.csv_data[col], prefix=col)
            self.csv_data = pd.concat([self.csv_data, dump_col], axis=1)
            self.csv_data.drop([col], axis=1, inplace=True)

    def set_missing_ages(self):
        age_df = self.csv_data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

        known_age = age_df[age_df.Age.notnull()].as_matrix()
        unknown_age = age_df[age_df.Age.isnull()].as_matrix()

        y = known_age[:, 0]
        X = known_age[:, 1:]

        if np.isnan(X).sum() > 0:
            X[np.isnan(X)] = np.nanmedian(X)

        if np.isinf(X).sum() > 0:
            X[np.isinf(X)] = np.nanmedian(X)
        if np.isneginf(X).sum() > 0:
            X[np.isneginf(X)] = np.nanmedian(X)


        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)

        predictedAges = rfr.predict(unknown_age[:, 1::])
        self.csv_data.loc[(self.csv_data.Age.isnull()), 'Age'] = predictedAges


    def set_cabin_type(self):
        self.csv_data.loc[(self.csv_data.Cabin.notnull()), 'Cabin'] = "Yes"
        self.csv_data.loc[(self.csv_data.Cabin.isnull()), 'Cabin'] = "No"

    def fit_logistic(self):
        train_df = self.csv_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_np = train_df.as_matrix()

        y= train_np[:, 0]
        X= train_np[:, 1:]

        print("fit data head:")
        print(X)

        clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        clf.fit(X, y)
        print(clf)
        return clf

if __name__ == '__main__':
    train_exp = Explore("dataset/train.csv", "PassengerId", "Survived", type_cols=["Pclass", "SibSp"])
    # exp.scan_type(type_cols=["Pclass", "SibSp", "Embarked", "Sex"])
    # exp.scan_serial(serial_cols=['Age'])
    # exp.scan_type_split(type_cols=["Pclass", "SibSp", "Embarked", "Sex"])
    # exp.scan_heatmap(drop_cols=['Name', 'Sex', 'Ticket', 'CaX[np.isnan(X)] = np.nanmedian(X)bin', 'Embarked'])
    # exp.test()
    train_exp.data_preprocess()

    print("train data head:")
    print(train_exp.csv_data.head(5))

    clf = train_exp.fit_logistic()

    test_exp = Explore("dataset/test.csv", "PassengerId", None, type_cols=["Pclass", "SibSp"])
    test_exp.data_preprocess()

    print("test data head:")
    print(test_exp.csv_data.head(5))

    predictions = clf.predict(test_exp.csv_data.iloc[:, 1:])
    result = pd.DataFrame({'PassengerId':test_exp.csv_data['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions.csv", index=False)



