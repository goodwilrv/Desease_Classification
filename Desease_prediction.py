# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:06:48 2019

@author: Dell 3450
"""

import os
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from collections import Counter
import matplotlib.pyplot as plt

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats
import statsmodels.tools.eval_measures as eval_mes
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
#from sklearn import tree
from sklearn.cluster import KMeans

from scipy import stats

np.set_printoptions(linewidth=200)

pd.get_option("display.max_rows")
pd.set_option("display.max_rows",200)
pd.set_option("display.max_colwidth",200)
pd.get_option("display.max_colwidth")


working_dir = "C:\\Users\\Dell 3450\\Desktop\\LnT_Documents\\BI_Assignment\\candidate_data\\candidate_data\\"

#def check_data_balanc(x_df):
    

def plot_distributions_hist(x_df,col_list):
    for col in col_list:
        x_df[[col]].hist(bins=10)
        
    

def plot_column(x_df,x_str):
    x_df[[x_str]].plot()
    
def format_str_To_Date(x_df,col_list,formStr):
    for col in col_list:
        x_df[col] = pd.to_datetime(x_df[col],format= formStr)
        x_df[col] = x_df[col].dt.date
        
    return x_df
    
    
def plotBoxPlotsByCancerType(x_df,cType):
    if(cType == 'cancer_type'):
         x_df[['radius_0','texture_0','perimeter_0','cancer_type']].boxplot(by='cancer_type')
         x_df[['radius_1','texture_1','perimeter_1','cancer_type']].boxplot(by='cancer_type')
         x_df[['radius_2','texture_2','perimeter_2','cancer_type']].boxplot(by='cancer_type')
         x_df[['diag_treat_diff','cancer_type']].boxplot(by='cancer_type')
         x_df[['age','cancer_type']].boxplot(by='cancer_type')
         
    elif(cType=='explanatory_vars'):
         temp_df = x_df.drop(['cancer_type','diagnose_date','treatment_date'],axis=1)
         temp_df.boxplot()
         plt.xticks(rotation=90)
        
     
     
#    salary_box_plot = x_df.boxplot(column=['salary','yearsExperience', 'milesFromMetropolis'])
#    x_df.describe()
#    salary_box_plot_Jobtype = x_df.boxplot(column=['salary'],by='jobType')
#    salary_box_plot_company = x_df.boxplot(column=['salary'],by='companyId')

def create_date_Diff_Var(x_df):
    x_df = format_str_To_Date(x_df,['diagnose_date','treatment_date'],'%Y-%m-%d') 
    x_df['diag_treat_diff'] = abs(x_df['treatment_date'] - x_df['diagnose_date'])
    x_df['diag_treat_diff'] = x_df['diag_treat_diff'].dt.days
    return x_df

def create_Imputation_Regression(x_df):
    numeric_columns =  ['radius_0','texture_0','perimeter_0','texture_1','radius_2','texture_2','perimeter_2','age','diag_treat_diff']

def apply_Liner_Regression(x_df):
    numeric_columns =  ['radius_0','texture_0','perimeter_0','texture_1','radius_2','texture_2','perimeter_2','age','diag_treat_diff']
    
    radius_1 = x_df[['radius_1']]
    X2 = sm.add_constant(x_df[numeric_columns])
    est = sm.OLS(radius_1, X2)
    est2 = est.fit()
    print(est2.summary())
    
    print("Type of p values",type(est2.pvalues))
    print("p values ",est2.pvalues)

  
def cluster_without_misVal_cols(x_df):
    
    inertia_list = list()
    no_cluster_list = list()
    for k in range(2,15):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x_df)
        centroids = kmeans.cluster_centers_
        print(centroids)
        print(kmeans.labels_)
        inertia_list.append(kmeans.inertia_)
        no_cluster_list.append(k)
    
    clust_cost_df = pd.DataFrame(data = {'No_Of_CLust':no_cluster_list,'Cost':inertia_list})
    clust_cost_df.to_csv("clust_cost_df.csv")
    clust_cost_df.plot()
    return clust_cost_df
    

#    plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#    
    
def impute_missing_Values_train(x_df):
    train_data_rad_1NoNAN_df = x_df[x_df.radius_1.notnull()]
    
    mean_radius_1_ct_0 = train_data_rad_1NoNAN_df.loc[train_data_rad_1NoNAN_df.cancer_type == 0,].radius_1.mean()
    mean_radius_1_ct_1 = train_data_rad_1NoNAN_df.loc[train_data_rad_1NoNAN_df.cancer_type == 1,].radius_1.mean()
    
    train_data_per1NoNAN_df = x_df[x_df.perimeter_1.notnull()]
    
    mean_periM_1_ct_0 = train_data_per1NoNAN_df.loc[train_data_per1NoNAN_df.cancer_type == 0,].perimeter_1.mean()
    mean_periM_1_ct_1 = train_data_per1NoNAN_df.loc[train_data_per1NoNAN_df.cancer_type == 1,].perimeter_1.mean()

    train_data_tx2Noxx_df = x_df.loc[x_df.texture_2 != 'xx',]
    
    train_data_tx2Noxx_df.texture_2 = train_data_tx2Noxx_df.texture_2.apply(float)
#    train_data_tx2Noxx_df.texture_2 = train_data_tx2Noxx_df.texture_2.apply(float)
    
    mean_text_2_ct_0 = train_data_tx2Noxx_df.loc[train_data_tx2Noxx_df.cancer_type == 0,].texture_2.mean()
    mean_text_2_ct_1 = train_data_tx2Noxx_df.loc[train_data_tx2Noxx_df.cancer_type == 1,].texture_2.mean()

    
    train_rad_1null_df = x_df[x_df.radius_1.isnull()]
#    train_rad_1null_df.loc[train_rad_1null_df.cancer_type == 0,'radius_1'] = mean_radius_1_ct_0
#    train_rad_1null_df.loc[train_rad_1null_df.cancer_type == 1,'radius_1'] = mean_radius_1_ct_1
#    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    train_rad_1null_df['radius_1'] = np.where(train_rad_1null_df.cancer_type == 0, mean_radius_1_ct_0, train_rad_1null_df.radius_1)
    train_rad_1null_df['radius_1'] = np.where(train_rad_1null_df.cancer_type == 1, mean_radius_1_ct_1, train_rad_1null_df.radius_1)
    train_rad_1null_df['texture_2'] = np.where(train_rad_1null_df.cancer_type == 0, mean_text_2_ct_0, train_rad_1null_df.texture_2)
    train_rad_1null_df['texture_2'] = np.where(train_rad_1null_df.cancer_type == 1, mean_text_2_ct_1, train_rad_1null_df.texture_2)
   
    
    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    
    train_peri_1null_df = x_df[x_df.perimeter_1.isnull()]
#    train_rad_1null_df.loc[train_rad_1null_df.cancer_type == 0,'radius_1'] = mean_radius_1_ct_0
#    train_rad_1null_df.loc[train_rad_1null_df.cancer_type == 1,'radius_1'] = mean_radius_1_ct_1
#    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    train_peri_1null_df['perimeter_1'] = np.where(train_peri_1null_df.cancer_type == 0, mean_periM_1_ct_0, train_peri_1null_df.perimeter_1)
    train_peri_1null_df['perimeter_1'] = np.where(train_peri_1null_df.cancer_type == 1, mean_periM_1_ct_1, train_peri_1null_df.perimeter_1)
    train_peri_1null_df['texture_2'] = np.where(train_peri_1null_df.cancer_type == 0, mean_text_2_ct_0, train_peri_1null_df.texture_2)
    train_peri_1null_df['texture_2'] = np.where(train_peri_1null_df.cancer_type == 1, mean_text_2_ct_1, train_peri_1null_df.texture_2)
   
    
    
    train_peri_1null_df.to_csv("train_peri_1null_df.csv")
    
    train_data_tx2xx_df = x_df.loc[(x_df.texture_2 == 'xx') & (x_df.perimeter_1.notnull() & (x_df.radius_1.notnull())),]
    
    train_data_tx2xx_df['texture_2'] = np.where(train_data_tx2xx_df.cancer_type == 0, mean_text_2_ct_0, train_data_tx2xx_df.texture_2)
    train_data_tx2xx_df['texture_2'] = np.where(train_data_tx2xx_df.cancer_type == 1, mean_text_2_ct_1, train_data_tx2xx_df.texture_2)
    train_data_tx2xx_df['radius_1'] = np.where(train_data_tx2xx_df.cancer_type == 0, mean_radius_1_ct_0, train_data_tx2xx_df.radius_1)
    train_data_tx2xx_df['radius_1'] = np.where(train_data_tx2xx_df.cancer_type == 1, mean_radius_1_ct_1, train_data_tx2xx_df.radius_1)
    train_data_tx2xx_df['perimeter_1'] = np.where(train_data_tx2xx_df.cancer_type == 0, mean_periM_1_ct_0, train_data_tx2xx_df.perimeter_1)
    train_data_tx2xx_df['perimeter_1'] = np.where(train_data_tx2xx_df.cancer_type == 1, mean_periM_1_ct_1, train_data_tx2xx_df.perimeter_1)
    
    
    train_data_tx2xx_df.to_csv("train_data_tx2xx_df.csv")
    
    replaced_values_df = pd.DataFrame(columns = list(train_data_df))
    replaced_values_df = replaced_values_df.append(train_rad_1null_df)
    replaced_values_df = replaced_values_df.append(train_peri_1null_df)
    replaced_values_df = replaced_values_df.append(train_data_tx2xx_df)
    
    x_df_rest = train_data_df[train_data_df.radius_1.notnull()]
    x_df_rest = x_df_rest[x_df_rest.perimeter_1.notnull()]
    x_df_rest = x_df_rest.loc[x_df_rest.texture_2 != 'xx',]
    
    x_return_df = pd.DataFrame(columns = list(train_data_df))
    x_return_df = x_return_df.append(x_df_rest)
    x_return_df = x_return_df.append(replaced_values_df)
    x_return_df.drop_duplicates(inplace=True)
#    x_return_df = x_return_df.append(train_data_tx2xx_df)
    return x_return_df


## this method calculates variable importance based on random forest and plots the variable importance
def calculatePlot_Variable_Importance(tr_x,tr_y):
    regr = RandomForestClassifier(max_depth=2,random_state=0,n_estimators=100)
    regr.fit(tr_x,tr_y.cancer_type.tolist())

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(tr_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(tr_x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    plt.xticks(range(tr_x.shape[1]), indices)
    plt.xlim([-1, tr_x.shape[1]])
    plt.show()
    


## this method divides the input dataframe into train and test data.
def create_train_test_df(x_df,text_percentage):
    x_df = x_df.loc[:, ~x_df.columns.str.contains('^Unnamed')]
    train_X = x_df.drop(['cancer_type'],axis=1)
    train_Y = x_df[['cancer_type']]   
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=text_percentage, random_state=42)

    return X_train,X_test,y_train,y_test


## this methods fits decision tree model with max. dept from 2,5,6,7,9
## and calculates corresponding RMSEs.
def apply_decision_tree_classification(x_df,imp_var):
    
    # Fit Decision Tree model
#    depth_list = [2,5,6,7,9]
#    final_col_rmse_df = pd.DataFrame(columns = {'Tree_Max_Depth','RMSE'})
#
    x_df = x_df.loc[:, ~x_df.columns.str.contains('^Unnamed')]
    x_train = x_df[imp_var]
    y_train = x_df.cancer_type
    
    y_train=y_train.astype('int')
    clf = DecisionTreeClassifier(random_state=0)
    DTScore = cross_val_score(clf, x_train, y_train, cv=10)
    print("Accuracy of decision tree with 10 cross validaion score=====>>>> ",str(DTScore))
    print("Mean and SD of Accuracy of decision tree with 10 cross validaion score=====>>>> " + "Mean: " + str(round(np.mean(DTScore),2)) + " Standard Deviation:" +str(round(np.std(DTScore),2)))
    
    
   
    clf.fit(x_tr_1, y_tr_1)
    
    y_pred = list(clf.predict(x_te_1))
#    y_pred = np.argmax(y_pred, axis=1)
    # Plot non-normalized confusion matrix
#    y_pred = (y_pred > 0.5)
    print("Type of y_te_1" ,type(y_te_1))
    y_true = y_te_1.cancer_type.tolist()
#    x2 = (x2 > 0.5)
#    class_names = [0,1]
    print("\n\n\n\n")
    print("Decision Tree ,Confusion Matrix:: ")
    print(confusion_matrix(y_true, y_pred))
    print("\n\n\n\n")
    
    return clf
    
#    for x in depth_list:
#        clf = DecisionTreeClassifier(random_state=0)
#        regr_1.fit(x_train_1, y_train_1)
#        y_1_pred = regr_1.predict(x_test_1)
#        
#        x1 = np.asanyarray(y_1_pred)
#        x2 = np.asanyarray(y_test)
#        this_rmse = np.sqrt(np.mean(np.square( x1 - x2)))
#        
#        this_rmse_df = pd.DataFrame(data={'Tree_Max_Depth':[x],'RMSE':[this_rmse]})
#   
#        final_col_rmse_df = final_col_rmse_df.append(this_rmse_df)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




def applyRandomForestClassification(x_df,imp_var):
    

    # Add noisy features
    random_state = np.random.RandomState(0)
    x_df1 = x_df[imp_var]
    X = x_df1.drop(['cancer_type'],axis=1)
    n_samples, n_features = X .shape
#    X = X.reset_index()
    
    y = x_df.cancer_type
#    y=y.reset_index()
    y=y.astype('int')
    
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    RFScore = cross_val_score(clf, X, y, cv=10)
    
    
    
    print("Accuracy of Random Forest with 10 cross validaion score=====>>>> ",str(RFScore))
    print("Mean and SD of Accuracy of Random Forest with 10 cross validaion score=====>>>> " + "Mean: " + str(round(np.mean(RFScore),2)) + " Standard Deviation:" +str(round(np.std(RFScore),2)))
    
   
    clf.fit(x_tr_1, y_tr_1)
    
    y_pred = list(clf.predict(x_te_1))
#    y_pred = np.argmax(y_pred, axis=1)
    # Plot non-normalized confusion matrix
#    y_pred = (y_pred > 0.5)
    
    y_true = y_te_1.cancer_type.tolist()
#    x2 = (x2 > 0.5)
#    class_names = [0,1]
    print("\n\n\n\n")
    print("Confusion Matrix:: ")
    print(confusion_matrix(y_true, y_pred))
    print("\n\n\n\n")
    
    return clf
    
def applyNeuralNetClassification(x_df,imp_var):
    random_state = np.random.RandomState(0)
    x_df1 = x_df[imp_var]
    X = x_df1.drop(['cancer_type'],axis=1)
    n_samples, n_features = X .shape
#    X = X.reset_index()
    
    y = x_df.cancer_type
#    y=y.reset_index()
    y=y.astype('int')
    
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

    MLPScore = cross_val_score(clf, X, y, cv=10)
    
    clf.fit(X, y)
    
    print("Accuracy of MLP with 10 cross validaion score=====>>>> ",str(MLPScore))
    print("Mean and SD of Accuracy of MLP with 10 cross validaion score=====>>>> " + "Mean: " + str(round(np.mean(MLPScore),2)) + " Standard Deviation:" +str(round(np.std(MLPScore),2)))
    
    
    clf.fit(x_tr_1, y_tr_1)
    
    y_pred = list(clf.predict(x_te_1))
#    y_pred = np.argmax(y_pred, axis=1)
    # Plot non-normalized confusion matrix
#    y_pred = (y_pred > 0.5)
    
    y_true = y_te_1.cancer_type.tolist()
#    x2 = (x2 > 0.5)
#    class_names = [0,1]
    print("\n\n\n\n")
    print("MLP Confusion Matrix:: ")
    print(confusion_matrix(y_true, y_pred))
    print("\n\n\n\n")
    
    return clf
    
def applyGradientBoostingClassification(x_df,imp_var):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

    
    random_state = np.random.RandomState(0)
    x_df1 = x_df[imp_var]
    X = x_df1.drop(['cancer_type'],axis=1)
    n_samples, n_features = X .shape
#    X = X.reset_index()
    
    y = x_df.cancer_type
#    y=y.reset_index()
    y=y.astype('int')

    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X)
#    X_test_scale = scaler.transform()
    
    X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = create_train_test_df(X_train_scale,0.20)
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in learning_rates:
        gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
        gb.fit(X_train_sub, y_train_sub)
        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
        print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
        print()
        

def applyLogisticRegressionClassification(x_df,imp_var):
    
    from sklearn.linear_model import LogisticRegression
    
    random_state = np.random.RandomState(0)
    x_df1 = x_df[imp_var]
    X = x_df1.drop(['cancer_type'],axis=1)
    n_samples, n_features = X .shape
#    X = X.reset_index()
    
    y = x_df.cancer_type
#    y=y.reset_index()
    y=y.astype('int')
    
    
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
    LRScore = cross_val_score(clf, X, y, cv=10)
    
    clf.fit(X, y)
    
#    print(clf.density())
    print("Accuracy of Logistic with 10 cross validaion score=====>>>> ",str(LRScore))
    print("Mean and SD of Accuracy of Logistic with 10 cross validaion score=====>>>> " + "Mean: " + str(round(np.mean(LRScore),2)) + " Standard Deviation:" +str(round(np.std(LRScore),2)))
    
    
    clf.fit(x_tr_1, y_tr_1)
    
    
    
    
    y_pred = list(clf.predict(x_te_1))
#    y_pred = np.argmax(y_pred, axis=1)
    # Plot non-normalized confusion matrix
#    y_pred = (y_pred > 0.5)
    
    y_true = y_te_1.cancer_type.tolist()
#    x2 = (x2 > 0.5)
#    class_names = [0,1]
    print("\n\n\n\n")
    print("Logistic Regression Confusion Matrix:: ")
    print(confusion_matrix(y_true, y_pred))
    print("\n\n\n\n")
     
    return clf


def final_test_impute_missing_Values_train(x_df):
    train_data_rad_1NoNAN_df = x_df[x_df.radius_1.notnull()]
    
    mean_radius_1_ct_0 = train_data_rad_1NoNAN_df.loc[train_data_rad_1NoNAN_df.Cluster == 0,].radius_1.mean()
    mean_radius_1_ct_1 = train_data_rad_1NoNAN_df.loc[train_data_rad_1NoNAN_df.Cluster == 1,].radius_1.mean()
    
    train_data_per1NoNAN_df = x_df[x_df.perimeter_1.notnull()]
    
    mean_periM_1_ct_0 = train_data_per1NoNAN_df.loc[train_data_per1NoNAN_df.Cluster == 0,].perimeter_1.mean()
    mean_periM_1_ct_1 = train_data_per1NoNAN_df.loc[train_data_per1NoNAN_df.Cluster == 1,].perimeter_1.mean()

    train_data_tx2Noxx_df = x_df.loc[x_df.texture_2 != 'xx',]
    
    train_data_tx2Noxx_df.texture_2 = train_data_tx2Noxx_df.texture_2.apply(float)
#    train_data_tx2Noxx_df.texture_2 = train_data_tx2Noxx_df.texture_2.apply(float)
    
    mean_text_2_ct_0 = train_data_tx2Noxx_df.loc[train_data_tx2Noxx_df.Cluster == 0,].texture_2.mean()
    mean_text_2_ct_1 = train_data_tx2Noxx_df.loc[train_data_tx2Noxx_df.Cluster == 1,].texture_2.mean()

    
    train_rad_1null_df = x_df[x_df.radius_1.isnull()]
#    train_rad_1null_df.loc[train_rad_1null_df.Cluster == 0,'radius_1'] = mean_radius_1_ct_0
#    train_rad_1null_df.loc[train_rad_1null_df.Cluster == 1,'radius_1'] = mean_radius_1_ct_1
#    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    train_rad_1null_df['radius_1'] = np.where(train_rad_1null_df.Cluster == 0, mean_radius_1_ct_0, train_rad_1null_df.radius_1)
    train_rad_1null_df['radius_1'] = np.where(train_rad_1null_df.Cluster == 1, mean_radius_1_ct_1, train_rad_1null_df.radius_1)
    train_rad_1null_df['texture_2'] = np.where(train_rad_1null_df.Cluster == 0, mean_text_2_ct_0, train_rad_1null_df.texture_2)
    train_rad_1null_df['texture_2'] = np.where(train_rad_1null_df.Cluster == 1, mean_text_2_ct_1, train_rad_1null_df.texture_2)
   
    
    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    
    train_peri_1null_df = x_df[x_df.perimeter_1.isnull()]
#    train_rad_1null_df.loc[train_rad_1null_df.Cluster == 0,'radius_1'] = mean_radius_1_ct_0
#    train_rad_1null_df.loc[train_rad_1null_df.Cluster == 1,'radius_1'] = mean_radius_1_ct_1
#    train_rad_1null_df.to_csv("train_rad_1null_df.csv")
    train_peri_1null_df['perimeter_1'] = np.where(train_peri_1null_df.Cluster == 0, mean_periM_1_ct_0, train_peri_1null_df.perimeter_1)
    train_peri_1null_df['perimeter_1'] = np.where(train_peri_1null_df.Cluster == 1, mean_periM_1_ct_1, train_peri_1null_df.perimeter_1)
    train_peri_1null_df['texture_2'] = np.where(train_peri_1null_df.Cluster == 0, mean_text_2_ct_0, train_peri_1null_df.texture_2)
    train_peri_1null_df['texture_2'] = np.where(train_peri_1null_df.Cluster == 1, mean_text_2_ct_1, train_peri_1null_df.texture_2)
   
    
    
    train_peri_1null_df.to_csv("train_peri_1null_df.csv")
    
    train_data_tx2xx_df = x_df.loc[(x_df.texture_2 == 'xx') & (x_df.perimeter_1.notnull() & (x_df.radius_1.notnull())),]
    
    train_data_tx2xx_df['texture_2'] = np.where(train_data_tx2xx_df.Cluster == 0, mean_text_2_ct_0, train_data_tx2xx_df.texture_2)
    train_data_tx2xx_df['texture_2'] = np.where(train_data_tx2xx_df.Cluster == 1, mean_text_2_ct_1, train_data_tx2xx_df.texture_2)
    train_data_tx2xx_df['radius_1'] = np.where(train_data_tx2xx_df.Cluster == 0, mean_radius_1_ct_0, train_data_tx2xx_df.radius_1)
    train_data_tx2xx_df['radius_1'] = np.where(train_data_tx2xx_df.Cluster == 1, mean_radius_1_ct_1, train_data_tx2xx_df.radius_1)
    train_data_tx2xx_df['perimeter_1'] = np.where(train_data_tx2xx_df.Cluster == 0, mean_periM_1_ct_0, train_data_tx2xx_df.perimeter_1)
    train_data_tx2xx_df['perimeter_1'] = np.where(train_data_tx2xx_df.Cluster == 1, mean_periM_1_ct_1, train_data_tx2xx_df.perimeter_1)
    
    
    train_data_tx2xx_df.to_csv("train_data_tx2xx_df.csv")
    
    replaced_values_df = pd.DataFrame(columns = list(final_test_data_df))
    replaced_values_df = replaced_values_df.append(train_rad_1null_df)
    replaced_values_df = replaced_values_df.append(train_peri_1null_df)
    replaced_values_df = replaced_values_df.append(train_data_tx2xx_df)
    
    x_df_rest = final_test_data_df[final_test_data_df.radius_1.notnull()]
    x_df_rest = x_df_rest[x_df_rest.perimeter_1.notnull()]
    x_df_rest = x_df_rest.loc[x_df_rest.texture_2 != 'xx',]
    
    x_return_df = pd.DataFrame(columns = list(final_test_data_df))
    x_return_df = x_return_df.append(x_df_rest)
    x_return_df = x_return_df.append(replaced_values_df)
    x_return_df.drop_duplicates(inplace=True)
#    x_return_df = x_return_df.append(train_data_tx2xx_df)
    return x_return_df



    
def apply_Liner_Regression(x_train_1,y_train_1,x_test_1,y_test_1):

    X2 = sm.add_constant(x_train_1)
    est = sm.OLS(y_train_1, X2)
    est2 = est.fit()
    print(est2.summary())
    
    print("Type of p values",type(est2.pvalues))
    print("p values ",est2.pvalues)

if __name__ == '__main__':
    
    os.chdir(working_dir)

    train_data_df = pd.read_csv("train_data.csv")
    final_test_data_df = pd.read_csv("test_data.csv")
    
    ## Check the balance of both cancer type category in terms of no. of observations.
    train_data_df.cancer_type.value_counts(normalize=True).reset_index().rename(index=str,columns={'index':'cancer_type','cancer_type':'Percentage'})
    
    
    
#    train_data_df = format_str_To_Date(train_data_df,['diagnose_date','treatment_date'],'%Y-%m-%d') 
#    train_data_df['diag_treat_diff'] = abs(train_data_df['treatment_date'] - train_data_df['diagnose_date'])
#    train_data_df['diag_treat_diff'] = train_data_df['diag_treat_diff'].dt.days
#    
    train_data_df = create_date_Diff_Var(train_data_df)
    final_test_data_df = create_date_Diff_Var(final_test_data_df)
    
    train_data_df.head()
    
    ## Check the mission percentage
    percent_missing = train_data_df.isnull().sum() * 100 / len(train_data_df)
    percent_missing_test = final_test_data_df.isnull().sum() * 100 / len(final_test_data_df)
    
   
    ## Remove the row with negative value for texture_0
    train_data_df = train_data_df.loc[train_data_df.texture_0 >=0,]
    
    
    
    
    
#    train_data_noNAN_df.shape
    
    train_data_tr2_xx_df = train_data_df.loc[train_data_df.texture_2 == 'xx',]
    train_data_tr2_xx_df.shape
    
    train_data_allNum_df = train_data_df[['radius_0','texture_0','perimeter_0','texture_1','radius_2','texture_2','perimeter_2','age','diag_treat_diff']]
    train_data_allNum_df = train_data_allNum_df.loc[train_data_allNum_df.texture_2 != 'xx',]
    
    clust_cost_df = cluster_without_misVal_cols(train_data_allNum_df)
    
    ## For final text data.
#    final_test_data_tr2_xx_df = final_test_data_df.loc[final_test_data_df.texture_2 == 'xx',]
#    final_test_data_tr2_xx_df.shape
    
    final_test_data_allNum_df = final_test_data_df[['radius_0','texture_0','perimeter_0','texture_1','radius_2','perimeter_2','age','diag_treat_diff']]
#    final_test_allNum_df = final_test_data_allNum_df.loc[final_test_data_allNum_df.texture_2 != 'xx',]
    
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(final_test_data_allNum_df)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print(kmeans.labels_)
    final_test_data_df = final_test_data_df.assign(Cluster = kmeans.labels_)
    
    final_test_imputed_data_df = final_test_impute_missing_Values_train(final_test_data_df)
    final_test_imputed_data_df.to_csv("final_test_imputed_data_df.csv")
    
    train_imputed_data_df = impute_missing_Values_train(train_data_df)
    

    
    numeric_cols = ['radius_0','texture_0','perimeter_0','radius_1','texture_1','perimeter_1','radius_2','texture_2','perimeter_2','age','diag_treat_diff']
    
    train_imputed_data_df[numeric_cols].describe()
    
    train_imputed_data_df['diag_treat_diff'] = train_imputed_data_df['diag_treat_diff'].apply(float)
    train_imputed_data_df['age'] = train_imputed_data_df['age'].apply(float)
    plotBoxPlotsByCancerType(train_imputed_data_df,'cancer_type')
    plotBoxPlotsByCancerType(train_imputed_data_df,'explanatory_vars')
    
#    train_data_df[train_data_df.id.apply(lambda x: x.isnumeric())]
    
    
    train_imputed_data_df[numeric_cols] = train_imputed_data_df[numeric_cols].astype(np.float)
    plot_distributions_hist(train_imputed_data_df,numeric_cols)
    
    ## If we try log transformation of some columns like age and daing_treat_diff
#    train_imputed_data_df['daig_treat_diff'] = train_imputed_data_df['daig_treat_diff']
    
    wol_train_data_df = train_imputed_data_df[numeric_cols][(np.abs(stats.zscore(train_imputed_data_df[numeric_cols])) < 3).all(axis=1)]
    
    
#   apply_linear
    
#    train_data_noRAD1NAN_df = train_data_df[train_data_df.radius_1.notnull()]
    
    ## Second strategy to impute the missing data.
#    train_data_allNum_df = train_data_df[['radius_0','texture_0','perimeter_0','radius_1','texture_1','perimeter_1','radius_2','texture_2','perimeter_2','age','diag_treat_diff']]
#    train_data_allNum_df = train_data_allNum_df.loc[train_data_allNum_df.texture_2 != 'xx',]
#    train_data_noNAN_df = train_data_df[train_data_df.radius_1.notnull()]
#    train_data_noNAN_df = train_data_noNAN_df[train_data_noNAN_df.perimeter_1.notnull()]
#    train_data_noNAN_df.shape
#    train_X = x_df.drop(['Store_Code','Last_Three_Quarter_Ave'],axis=1)
#    train_Y = x_df[['Last_Three_Quarter_Ave']]
#    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=text_percentage, random_state=42)
#      
#    apply_Liner_Regression(train_data_noNAN_df)
    
#    train_imputed_data_df['daig_treat_diff']  = (1+train_imputed_data_df['daig_treat_diff'] )/2 # (-1,1] -> (0,1]
    train_imputed_data_df['log_diag_treat_diff']  = np.log(train_imputed_data_df['diag_treat_diff'])
    train_imputed_data_df['log_age']  = np.log(train_imputed_data_df['age'])
    train_imputed_data_df['log_perimeter_2']  = np.log(train_imputed_data_df['perimeter_2'])
    train_imputed_data_df['log_texture_1']  = np.log(train_imputed_data_df['texture_1'])
    
    numeric_cols.extend(['log_diag_treat_diff','log_age','log_perimeter_2','log_texture_1'])
    plot_distributions_hist(train_imputed_data_df,numeric_cols)
    
    
    
    ######## Divide the train data into test and train
    selected_columns = ['radius_0','texture_0','perimeter_0','radius_1','log_texture_1','perimeter_1','radius_2','texture_2','perimeter_2','age','log_diag_treat_diff','cancer_type']
    
    x_train, x_test, y_train, y_test =  create_train_test_df(train_imputed_data_df[selected_columns],0.20)
    ## Calculate variable importance based on Variable importance statistics of random forest.
    calculatePlot_Variable_Importance(x_train,y_train)
    
    important_Variables = [list(x_train)[i] for i in [0,2,3,5,6,7]]
    
    important_Variables_rf = important_Variables
    important_Variables_rf.extend(['cancer_type'])
    
    ###########################3 Start the Modelling with only important variables #############33
    
    x_tr_1, x_te_1, y_tr_1, y_te_1 =  create_train_test_df(train_imputed_data_df[important_Variables_rf],0.20)
    y_tr_1=y_tr_1.astype('int')
    
    dt_model = apply_decision_tree_classification(train_imputed_data_df,important_Variables_rf)
    
    
    
    rf_model = applyRandomForestClassification(train_imputed_data_df,important_Variables_rf)
    
    
#    plot_confusion_matrix(x2, y_pred, classes=class_names,title='Confusion matrix, without normalization')

    
    ################# Final Prediction #########################################3
    mlp_model = applyNeuralNetClassification(train_imputed_data_df,important_Variables_rf)
    
    lr_model = applyLogisticRegressionClassification(train_imputed_data_df,important_Variables_rf)
    
    
    final_test_tr = final_test_imputed_data_df[list(x_tr_1)]
    final_pred_rf = list(rf_model.predict(final_test_tr))
    pd.DataFrame(data ={'cancer_type':final_pred_rf}).to_csv("final_prediction_rf.csv")
    
    final_pred_dt = list(dt_model.predict(final_test_tr))
    pd.DataFrame(data ={'cancer_type':final_pred_dt}).to_csv("final_prediction_dt.csv")
    
    final_pred_mlp = list(mlp_model.predict(final_test_tr))
    pd.DataFrame(data ={'cancer_type':final_pred_mlp}).to_csv("final_prediction_mlp.csv")
    
    final_pred_lr = list(lr_model.predict(final_test_tr))
    pd.DataFrame(data ={'cancer_type':final_pred_lr}).to_csv("final_prediction_lr.csv")
    
#    from sklearn import metrics
    
#    important_Variables.extend(['cancer_type'])
#    x_train, x_test, y_train, y_test =  create_train_test_df(train_imputed_data_df[important_Variables],0.20)
    ## Calculate variable importance based on Variable importance statistics of random forest.
   
#    y_train=y_train.astype('int')
#    clf = DecisionTreeClassifier(random_state=0)
#    clf = clf.fit(x_train,y_train)
#    
#    cross_val_score(clf, x_train, y_train, cv=10)
    
#    y_pred = clf.predict(x_test)
#    
#    metrics.accuracy_score(y_test, y_pred)
    
    
    ################Linear Regression ##########################333
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    
    X_reg = x_tr_1[['perimeter_1']]
    Y_reg = x_tr_1[['radius_2']]
    
    reg = LinearRegression().fit(X_reg,Y_reg )
    reg.score(X_reg, Y_reg)
    
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(X_reg, Y_reg)):
     reg.fit(X_reg.iloc[train,:], Y_reg.iloc[train,:])
     score = reg.score(X_reg.iloc[test,:], Y_reg.iloc[test,:])
     scores.append(score)
    print(scores)
    
    
    
    y_pred = reg.predict(x_te_1)
    
    
    
    


    


