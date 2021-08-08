from rest_framework.generics import ListCreateAPIView,RetrieveUpdateDestroyAPIView
from api.churnapi.models import UserRegistrationModel
from api.churnapi.serializations import UserRegisterSerializer
from django.http import HttpResponse
from django.conf import settings
from rest_framework import filters
import io
from rest_framework.parsers import  JSONParser
# import necccessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import json
class CustomerInfo(ListCreateAPIView):
    queryset = UserRegistrationModel.objects.all()
    serializer_class = UserRegisterSerializer
    #ilter_backends = (filters.BaseFilterBackend,)

    '''
    def get_queryset(self):
        #name = self.request.GET.get('loginid') #self.kwargs('loginid')
        req = self.request.body
        if len(req)==0:
            return UserRegistrationModel.objects.all()
        stream = io.BytesIO(req)
        pdata = JSONParser().parse(stream)
        loginid =pdata['loginid']
        pswd = pdata['password']
        if loginid is not None:
            print('Login ID:',loginid,' Password ',pswd )
            return UserRegistrationModel.objects.filter(loginid=loginid,password=pswd)
        else:
            return UserRegistrationModel.objects.all()
        return UserRegistrationModel.objects.all()
    '''


class CustomerCrudInfo(RetrieveUpdateDestroyAPIView):
    queryset = UserRegistrationModel.objects.all()
    serializer_class = UserRegisterSerializer
    lookup_field = 'id'

from django.views.generic import View
class DatasetAPICallFirstDF(View):
    def get(self,request,*args,**kwargs):
        path = settings.MEDIA_ROOT + "\\" + "dataset1.csv"
        df = pd.read_csv(path)
        # Remove customerID column because it is useless
        df.drop("customerID", axis=1, inplace=True)
        df.dtypes
        df.TotalCharges.values
        pd.to_numeric(df.TotalCharges, errors="coerce").isnull()
        df[pd.to_numeric(df.TotalCharges, errors="coerce").isnull()]
        df["TotalCharges"][488]
        df1 = df[df.TotalCharges != ' ']
        print(df1.shape)
        df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
        df1.TotalCharges.dtypes

        #Churn Prediction Graph
        MonthlyCharges_churn_no = df1[df1.Churn == "No"].MonthlyCharges
        MonthlyCharges_churn_yes = df1[df1.Churn == "Yes"].MonthlyCharges
        plt.xlabel("MonthlyCharges")
        plt.ylabel("No of Customers")
        plt.title("Customer Churn Prediction")
        plt.hist([MonthlyCharges_churn_yes, MonthlyCharges_churn_no], rwidth=0.95, color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
        plt.legend()
        #plt.show()

        #Tenure Graph
        tanure_churn_no = df1[df1.Churn == "No"].tenure
        tanure_churn_yes = df1[df1.Churn == "Yes"].tenure
        plt.xlabel("Tenure")
        plt.ylabel("No of Customers")
        plt.title("Customer Churn Prediction")
        plt.hist([tanure_churn_yes, tanure_churn_no], rwidth=0.95, color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
        plt.legend()
        #plt.show()
        for i in df1.columns:
            if df1[i].dtypes == "object":
                print(f'{i}: {df1[i].unique()}')

        df1.replace('No internet service', 'No', inplace=True)
        df1.replace('No phone service', 'No', inplace=True)
        # Replace Value of " Yes" and " No" to 1 and 0
        yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                          'Churn']
        for i in yes_no_columns:
            df1[i].replace({"Yes": 1, "No": 0}, inplace=True)

        for i in df1.columns:
            print(df1[i].unique())

        df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
        df1.gender.unique()
        df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
        df2.columns
        df2.head()
        cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
        df2.head(5)

        for col in df2.columns:
            print(f'{col} : {df2[col].unique()}')

        X = df2.drop('Churn', axis='columns')
        y = df2['Churn']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        '''
        import tensorflow as tf
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Dense(26, input_shape=(26,), activation="relu"),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        # opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=100)
        model.evaluate(X_test, y_test)
        # Let do it for test data
        y_predict = model.predict(X_test)
        y_predict[:5]
        y_p = []
        for i in y_predict:
            if i >= 0.5:
                y_p.append(1)
            else:
                y_p.append(0)
        
        print(classification_report(y_test, y_p))
        #import seaborn as sns
        #confusion_metrix = tf.math.confusion_matrix(labels=y_test, predictions=y_p)
        #plt.figure(figsize=(10, 7))
        #sns.heatmap(confusion_metrix, annot=True, fmt='d')
        #plt.xlabel('Predicted')
        #plt.ylabel('Truth')
        #plt.show()
        dnn_accuracy = accuracy_score(y_test, y_p)
        print('DNN accuracy:',dnn_accuracy)
        '''
        from sklearn.linear_model import LogisticRegression
        lg = LogisticRegression(penalty='l1',solver='liblinear')
        lg.fit(X_train,y_train)
        y_pred = lg.predict(X_test)
        lg_accuracy = accuracy_score(y_test,y_pred)
        lg_precision = precision_score(y_test, y_pred)
        lg_recall = recall_score(y_test, y_pred)
        lg_f1score = f1_score(y_test, y_pred)
        lg_roc = roc_auc_score(y_test, y_pred)
        dict_lg = {'lg_accuracy':lg_accuracy,'lg_precision':lg_precision,'lg_recall':lg_recall,'lg_f1score':lg_f1score,'lg_roc':lg_roc}
        print('Logistic Results ',dict_lg)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_features=15, max_leaf_nodes=20)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_pred)
        dt_precision = precision_score(y_test, y_pred)
        dt_recall = recall_score(y_test, y_pred)
        dt_f1score = f1_score(y_test, y_pred)
        dt_roc = roc_auc_score(y_test, y_pred)
        dict_dt = {'dt_accuracy': dt_accuracy, 'dt_precision': dt_precision, 'dt_recall': dt_recall,
                   'dt_f1score': dt_f1score, 'dt_roc': dt_roc}
        print('Decision Tree Results ', dict_dt)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, max_features=15, max_leaf_nodes=25)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)
        rf_precision = precision_score(y_test, y_pred)
        rf_recall = recall_score(y_test, y_pred)
        rf_f1score = f1_score(y_test, y_pred)
        rf_roc = roc_auc_score(y_test, y_pred)
        dict_rf = {'rf_accuracy': rf_accuracy, 'rf_precision': rf_precision, 'rf_recall': rf_recall,
                   'rf_f1score': rf_f1score, 'rf_roc': rf_roc}
        print('RandomForest  Results ', dict_rf)
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier(n_estimators=60, learning_rate=1.0)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        ada_accuracy = accuracy_score(y_test, y_pred)
        ada_precision = precision_score(y_test, y_pred)
        ada_recall = recall_score(y_test, y_pred)
        ada_f1score = f1_score(y_test, y_pred)
        ada_roc = roc_auc_score(y_test, y_pred)
        dict_ada = {'ada_accuracy': ada_accuracy, 'ada_precision': ada_precision, 'ada_recall': ada_recall,
                   'ada_f1score': ada_f1score, 'ada_roc': ada_roc}
        print('Adaboost  Results ', dict_ada)

        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(10,))
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, y_pred)
        mlp_precision = precision_score(y_test, y_pred)
        mlp_recall = recall_score(y_test, y_pred)
        mlp_f1score = f1_score(y_test, y_pred)
        mlp_roc = roc_auc_score(y_test, y_pred)
        dict_mlp = {'mlp_accuracy': mlp_accuracy, 'mlp_precision': mlp_precision, 'mlp_recall': mlp_recall,
                    'mlp_f1score': mlp_f1score, 'mlp_roc': mlp_roc}
        print('Multi layer Perceptron Results ', dict_mlp)
        dataset = df2.to_dict()
        result = {'dataset':dataset,'dict_lg':dict_lg,'dict_dt':dict_dt,'dict_rf':dict_rf,'dict_ada':dict_ada,'dict_mlp':dict_mlp}
        json_object = json.dumps(result)
        return HttpResponse(json_object, content_type = 'application/json')



class DatasetAPICallSecondDF(View):
    def get(self,request,*args,**kwargs):
        path = settings.MEDIA_ROOT + "\\" + "dataset2.csv"
        df = pd.read_csv(path)
        from sklearn import preprocessing
        df['churn'].replace({'no': 0, 'yes': 1}, inplace=True)
        #y_True = df["churn"][df["churn"] == True]
        #print("Churn Percentage = " + str((y_True.shape[0] / df["churn"].shape[0]) * 100))
        # Discreet value integer encoder
        label_encoder = preprocessing.LabelEncoder()
        # State is string and we want discreet integer values
        df['state'] = label_encoder.fit_transform(df['state'])
        df['international_plan'] = label_encoder.fit_transform(df['international_plan'])
        df['voice_mail_plan'] = label_encoder.fit_transform(df['voice_mail_plan'])

        # print (df['Voice mail plan'][:4])
        print(df.dtypes)
        y = df['churn']
        y.size
        # df = df.drop(["Id","Churn"], axis = 1, inplace=True)
        df.drop(['area_code',"churn"], axis=1, inplace=True)
        df.head(3)
        X = df
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

        '''
        def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
            stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
            y_pred = y.copy()
            # ii -> train
            # jj -> test indices
            for ii, jj in stratified_k_fold:
                X_train, X_test = X[ii], X[jj]
                y_train = y[ii]
                clf = clf_class(**kwargs)
                clf.fit(X_train, y_train)
                y_pred[jj] = clf.predict(X_test)
            return y_pred
            '''
        #print(X)
        #print('####################################')
        #print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        from sklearn.linear_model import LogisticRegression
        lg = LogisticRegression(penalty='l1', solver='liblinear')
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
        lg_accuracy = accuracy_score(y_test, y_pred)
        lg_precision = precision_score(y_test, y_pred)
        lg_recall = recall_score(y_test, y_pred)
        lg_f1score = f1_score(y_test, y_pred)
        lg_roc = roc_auc_score(y_test, y_pred)
        dict_lg = {'lg_accuracy': lg_accuracy, 'lg_precision': lg_precision, 'lg_recall': lg_recall,
                   'lg_f1score': lg_f1score, 'lg_roc': lg_roc}
        print('Logistic Results ', dict_lg)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_features=15, max_leaf_nodes=20)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_pred)
        dt_precision = precision_score(y_test, y_pred)
        dt_recall = recall_score(y_test, y_pred)
        dt_f1score = f1_score(y_test, y_pred)
        dt_roc = roc_auc_score(y_test, y_pred)
        dict_dt = {'dt_accuracy': dt_accuracy, 'dt_precision': dt_precision, 'dt_recall': dt_recall,
                   'dt_f1score': dt_f1score, 'dt_roc': dt_roc}
        print('Decision Tree Results ', dict_dt)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, max_features=15, max_leaf_nodes=25)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)
        rf_precision = precision_score(y_test, y_pred)
        rf_recall = recall_score(y_test, y_pred)
        rf_f1score = f1_score(y_test, y_pred)
        rf_roc = roc_auc_score(y_test, y_pred)
        dict_rf = {'rf_accuracy': rf_accuracy, 'rf_precision': rf_precision, 'rf_recall': rf_recall,
                   'rf_f1score': rf_f1score, 'rf_roc': rf_roc}
        print('RandomForest  Results ', dict_rf)
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier(n_estimators=60, learning_rate=1.0)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        ada_accuracy = accuracy_score(y_test, y_pred)
        ada_precision = precision_score(y_test, y_pred)
        ada_recall = recall_score(y_test, y_pred)
        ada_f1score = f1_score(y_test, y_pred)
        ada_roc = roc_auc_score(y_test, y_pred)
        dict_ada = {'ada_accuracy': ada_accuracy, 'ada_precision': ada_precision, 'ada_recall': ada_recall,
                    'ada_f1score': ada_f1score, 'ada_roc': ada_roc}
        print('Adaboost  Results ', dict_ada)

        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(10,))
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, y_pred)
        mlp_precision = precision_score(y_test, y_pred)
        mlp_recall = recall_score(y_test, y_pred)
        mlp_f1score = f1_score(y_test, y_pred)
        mlp_roc = roc_auc_score(y_test, y_pred)
        dict_mlp = {'mlp_accuracy': mlp_accuracy, 'mlp_precision': mlp_precision, 'mlp_recall': mlp_recall,
                    'mlp_f1score': mlp_f1score, 'mlp_roc': mlp_roc}
        print('Multi layer Perceptron Results ', dict_mlp)
        dataset = df.to_dict()
        result = {'dataset': dataset, 'dict_lg': dict_lg, 'dict_dt': dict_dt, 'dict_rf': dict_rf, 'dict_ada': dict_ada,
                  'dict_mlp': dict_mlp}
        json_object = json.dumps(result)
        return HttpResponse(json_object, content_type='application/json')



class DatasetAPICallThirdDF(View):
    def get(self,request,*args,**kwargs):
        path1 = settings.MEDIA_ROOT + "\\" + "cell2celltrain.csv"
        path2 = settings.MEDIA_ROOT + "\\" + "cell2cellholdout.csv"
        train = pd.read_csv(path1)
        test = pd.read_csv(path2)
        train.info()
        train[0:10]

        # Churn : Yes:1 , No:0
        Churn = {'Yes': 1, 'No': 0}

        # traversing through dataframe
        # values where key matches
        train.Churn = [Churn[item] for item in train.Churn]
        #print(train)
        print("Any missing sample in training set:", train.isnull().values.any())
        print("Any missing sample in test set:", test.isnull().values.any(), "\n")
        # for column
        # train['MonthlyRevenue'].fillna((train['MonthlyRevenue'].median()), inplace=True)
        # for column
        train['MonthlyRevenue'] = train['MonthlyRevenue'].replace(np.nan, 0)

        # for whole dataframe
        train = train.replace(np.nan, 0)

        # inplace
        train.replace(np.nan, 0, inplace=True)

        print(train)

        # for column
        # train['MonthlyMinutes'].fillna((train['MonthlyMinutes'].median()), inplace=True)
        train['MonthlyMinutes'] = train['MonthlyMinutes'].replace(np.nan, 0)

        # for whole dataframe
        train = train.replace(np.nan, 0)

        # inplace
        train.replace(np.nan, 0, inplace=True)

        print(train)

        # for column
        # train['TotalRecurringCharge'].fillna((train['TotalRecurringCharge'].median()), inplace=True)
        train['TotalRecurringCharge'] = train['TotalRecurringCharge'].replace(np.nan, 0)

        # for whole dataframe
        train = train.replace(np.nan, 0)

        # inplace
        train.replace(np.nan, 0, inplace=True)

        print(train)

        # for column
        # train['DirectorAssistedCalls'].fillna((train['DirectorAssistedCalls'].median()), inplace=True)
        train['DirectorAssistedCalls'] = train['DirectorAssistedCalls'].replace(np.nan, 0)

        # for whole dataframe
        train = train.replace(np.nan, 0)

        # inplace
        train.replace(np.nan, 0, inplace=True)

        print(train)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        def FunLabelEncoder(df):
            for c in df.columns:
                if df.dtypes[c] == object:
                    le.fit(df[c].astype(str))
                    df[c] = le.transform(df[c].astype(str))
            return df

        train = FunLabelEncoder(train)
        train.info()
        train.iloc[235:300, :]
        test = FunLabelEncoder(test)
        test.info()
        test.iloc[235:300, :]
        test = test.drop(columns=['Churn'],

                         axis=1)
        test = test.dropna(how='any')
        print(test.shape)

        # Frequency distribution of classes"
        train_outcome = pd.crosstab(index=train["Churn"],  # Make a crosstab
                                    columns="count")  # Name the count column

        train_outcome

        # Distribution of Churn
        train.Churn.value_counts()[0:30].plot(kind='bar')
        #plt.show()
        train = train[
            ['CustomerID', 'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls',
             'OverageMinutes',
             'RoamingCalls', 'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls', 'BlockedCalls',
             'UnansweredCalls', 'CustomerCareCalls',
             'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 'PeakCallsInOut', 'OffPeakCallsInOut',
             'DroppedBlockedCalls', 'CallForwardingCalls'
                , 'CallWaitingCalls', 'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'ServiceArea', 'Handsets',
             'HandsetModels',
             'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable',
             'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings',
             'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'RetentionCalls', 'RetentionOffersAccepted',
             'NewCellphoneUser',
             'NotNewCellphoneUser', 'ReferralsMadeBySubscriber', 'IncomeGroup', 'OwnsMotorcycle',
             'AdjustmentsToCreditRating',
             'HandsetPrice', 'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus',
             'Churn']]  # Subsetting the data
        cor = train.corr()  # Calculate the correlation of the above variables
        #sns.heatmap(cor, square=True)  # Plot the correlation as heat map
        from sklearn.model_selection import train_test_split
        y = train['Churn']
        X = train.drop(columns=['Churn'])
        #X_train, X_test, y_train, y_tes = train_test_split(X, Y, test_size=0.3, random_state=9)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        from sklearn.linear_model import LogisticRegression
        lg = LogisticRegression(penalty='l1', solver='liblinear')
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
        lg_accuracy = accuracy_score(y_test, y_pred)
        lg_precision = precision_score(y_test, y_pred)
        lg_recall = recall_score(y_test, y_pred)
        lg_f1score = f1_score(y_test, y_pred)
        lg_roc = roc_auc_score(y_test, y_pred)
        dict_lg = {'lg_accuracy': lg_accuracy, 'lg_precision': lg_precision, 'lg_recall': lg_recall,
                   'lg_f1score': lg_f1score, 'lg_roc': lg_roc}
        print('Logistic Results ', dict_lg)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_features=15, max_leaf_nodes=20)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        dt_accuracy = accuracy_score(y_test, y_pred)
        dt_precision = precision_score(y_test, y_pred)
        dt_recall = recall_score(y_test, y_pred)
        dt_f1score = f1_score(y_test, y_pred)
        dt_roc = roc_auc_score(y_test, y_pred)
        dict_dt = {'dt_accuracy': dt_accuracy, 'dt_precision': dt_precision, 'dt_recall': dt_recall,
                   'dt_f1score': dt_f1score, 'dt_roc': dt_roc}
        print('Decision Tree Results ', dict_dt)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, max_features=15, max_leaf_nodes=25)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)
        rf_precision = precision_score(y_test, y_pred)
        rf_recall = recall_score(y_test, y_pred)
        rf_f1score = f1_score(y_test, y_pred)
        rf_roc = roc_auc_score(y_test, y_pred)
        dict_rf = {'rf_accuracy': rf_accuracy, 'rf_precision': rf_precision, 'rf_recall': rf_recall,
                   'rf_f1score': rf_f1score, 'rf_roc': rf_roc}
        print('RandomForest  Results ', dict_rf)
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier(n_estimators=60, learning_rate=1.0)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        ada_accuracy = accuracy_score(y_test, y_pred)
        ada_precision = precision_score(y_test, y_pred)
        ada_recall = recall_score(y_test, y_pred)
        ada_f1score = f1_score(y_test, y_pred)
        ada_roc = roc_auc_score(y_test, y_pred)
        dict_ada = {'ada_accuracy': ada_accuracy, 'ada_precision': ada_precision, 'ada_recall': ada_recall,
                    'ada_f1score': ada_f1score, 'ada_roc': ada_roc}
        print('Adaboost  Results ', dict_ada)

        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(10,))
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, y_pred)
        mlp_precision = precision_score(y_test, y_pred)
        mlp_recall = recall_score(y_test, y_pred)
        mlp_f1score = f1_score(y_test, y_pred)
        mlp_roc = roc_auc_score(y_test, y_pred)
        dict_mlp = {'mlp_accuracy': mlp_accuracy, 'mlp_precision': mlp_precision, 'mlp_recall': mlp_recall,
                    'mlp_f1score': mlp_f1score, 'mlp_roc': mlp_roc}
        print('Multi layer Perceptron Results ', dict_mlp)
        dataset = test.to_dict()
        result = {'dataset': dataset, 'dict_lg': dict_lg, 'dict_dt': dict_dt, 'dict_rf': dict_rf, 'dict_ada': dict_ada,
                  'dict_mlp': dict_mlp}
        json_object = json.dumps(result)
        return HttpResponse(json_object, content_type='application/json')

