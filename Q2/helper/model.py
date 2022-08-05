
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.linear_model import ElasticNet, LinearRegression
# from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    recall_score, f1_score, accuracy_score, balanced_accuracy_score,
    mean_squared_error, auc, classification_report
)
from helper.metrics import plot_confusion_matrix
from helper.pickle_utils import save_pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# def train_model()
def train_model_CV(schema, df_train, df_test, cv=3, scoring='accuracy', save_model=False):
    
    model_name = schema['model_name']
    classifier = schema['model_obj']()
    vectorizer = schema['vec_obj']
    grid_params = schema['grid_params']
    model_path = schema['model_path']
    plot_path = schema['plot_path']

    print('=*'*6,'Training for', model_name,'=*'*6)
    
    results={}
    
    x_train = df_train['tweet_text']
    x_test = df_test['tweet_text']
    y_train = df_train['topic']
    y_test = df_test['topic']

    ## Smaller Subset for GridSearch ##
    # df_train_subset=df_train.sample(frac=0.20)
    # x_train_subset=df_train_subset[FEATURE_COLS]
    # y_train_subset=df_train_subset[OUT_COL].values.ravel()
    
    # x_train=df_train[FEATURE_COLS]
    # x_test=df_test[FEATURE_COLS]
    # y_train=df_train[OUT_COL].values.ravel()
    # y_test=df_test[OUT_COL].values.ravel()

    pipeline = Pipeline([
        ('vect', vectorizer),
        ('chi', SelectKBest(chi2, k=1200)),
        ('clf', classifier)
        ])

    grid_search=GridSearchCV(pipeline, grid_params, cv=cv,scoring=scoring, n_jobs=-1, refit=True)
    
    #%time 
    grid_search.fit(x_train, y_train)
    best_params=grid_search.best_params_
    
    print('Best Parameters:', best_params)
    
    Model=pipeline.set_params(**best_params)

    #%time 
    Model.fit(x_train, y_train)
    y_pred=Model.predict(x_test)
    # y_proba=Model.predict_proba(x_test)
    # score=np.sqrt(mean_squared_error(y_test,y_pred))
    # precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:,0])
    # f1 = f1_score(y_test, y_pred,)
    
    # results['Score_F1']=np.round(f1,3)
    # class_report = classification_report(y_test, y_pred)
    score = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = plot_confusion_matrix(y_test, y_pred, return_matrix=True, save_fig_path=plot_path)

    results['Model_Name']=model_name
    results['Best_Params']=best_params
    results['Confusion_Matrix'] = conf_matrix
    results['Score'] = score
    
    print('{}, Score : {}'.format(model_name, np.round(score,3)))
    
    if save_model:
        try:
            save_pickle(Model, model_path)
            print(f'INFO : Model saved to {model_path}')
        except:
            print(f'ERROR : Failed to save Model to {model_path}')
    print('=*'*20)
    return Model, results