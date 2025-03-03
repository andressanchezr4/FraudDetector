# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andres.sanchez
"""
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
from xgboost import XGBClassifier

class SuperLearner(object):
    
    def __init__(self, train_x, train_y,
                 model_list = [], meta_model = LogisticRegression(), 
                 ):
       
        self.train_y = train_y.to_numpy().ravel()
        if not isinstance(train_x, np.ndarray):
            self.train_x = train_x.to_numpy()
        else:
            self.train_x = train_x
        if np.issubdtype(self.train_y.dtype, np.number):
            self.regression = True
        else:
            self.regression = False
          
        self.meta_model = meta_model
        if len(model_list) == 0:
            self.get_base_models()
        else:
            self.base_models = model_list
     
        self.n_cv = len(self.base_models)
        
        rows_per_part = len(self.train_x) // self.n_cv  
        self.train_x = self.train_x[:rows_per_part * self.n_cv]
        self.train_y = self.train_y[:rows_per_part * self.n_cv]
        
        self.oof_X = 'trained_oofX'
        self.oof_y = 'trained_oofy'
        
    def get_base_models(self, random_state=1234):
        
        self.base_models_raw= self.base_models = [
            LogisticRegression(),
            DecisionTreeClassifier(random_state=random_state),
            SVC(probability=True),
            GaussianNB(),
            KNeighborsClassifier(),
            AdaBoostClassifier(random_state=random_state),
            BaggingClassifier(n_estimators=200, random_state=random_state),
            RandomForestClassifier(n_estimators=300, random_state=random_state),
            ExtraTreesClassifier(n_estimators=200, random_state=random_state),
            XGBClassifier(
                random_state=random_state,
                n_estimators=1000,
                objective="binary:logistic",
                eval_metric="logloss",
                          ),
            ]
        
    def train_each_model_on_fold(self):
        meta_X, meta_y = [], []
        
        kfold = KFold(n_splits=self.n_cv, shuffle=True)
        	
        for n, (train_ix, test_ix) in enumerate(kfold.split(self.train_x)):
            fold_yhats = []
            train_X, test_X = self.train_x[train_ix], self.train_x[test_ix]
            train_y, test_y = self.train_y[train_ix], self.train_y[test_ix]
            meta_y.extend(test_y)
        		
            for m, model in enumerate(self.base_models):
                model.fit(train_X, train_y)
                yhat = model.predict_proba(test_X)[:, 1]
                fold_yhats.extend(yhat)
                print(f'Done with {model.__class__()}')
                print(f'{m+1}/{len(self.base_models)} models')
                print(f'-------------{n+1}/{self.n_cv} CVs-------------')
            
            meta_X.append(np.hstack(fold_yhats))

        self.oof_X = np.vstack(meta_X).transpose()
        self.oof_y = np.asarray(meta_y)
    
    def fit_meta_model(self):
        self.meta_model.fit(self.oof_X, self.oof_y)
               
    def fit_base_models(self):
        for n, model in enumerate(self.base_models):
            model.fit(self.train_x, self.train_y)
            print(f'{n+1}/{len(self.base_models)} Base models trained')
    
    def evaluate_models(self, val_x, val_y):
        self.all_pr = {}
        print('Each model individual PR')
        print('-'*20)
        for model in self.base_models:
            y_pred = model.predict_proba(val_x)[:, 1]
            precision, recall, _ = precision_recall_curve(val_y, y_pred)
            pr = auc(recall, precision)
            
            self.all_pr[model] = pr
            print(f'{model.__class__.__name__} PR: {pr:.3f}')
        
    def fit(self):
    
        self.train_each_model_on_fold()
        self.fit_base_models()
        self.fit_meta_model()
    
    def alreadytrainedoof(self):
        self.fit_base_models()
        self.fit_meta_model()
        
    def predict(self, X):
        
        self.meta_X = list()
        
        for model in self.base_models:
            y_pred = model.predict_proba(X)[:, 1]
            self.meta_X.append(y_pred.reshape(len(y_pred),1))
      	
        self.meta_X = np.hstack(self.meta_X)
        
        return self.meta_model.predict_proba(self.meta_X)[:, 1]
        
    

 