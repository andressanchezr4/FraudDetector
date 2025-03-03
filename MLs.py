# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andres.sanchez
"""

from sklearn.metrics import precision_recall_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay
import os
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MLsPipeline(object):
    
    def __init__(self, train_x, train_y, val_x, val_y, 
                 path2figures,  n_cv = 8, random_state=1234):
        self.train_x = train_x
        self.val_x = val_x
        self.n_cv = n_cv
        self.random_state = random_state
        
        self.train_y = train_y.to_numpy().ravel()
        self.val_y = val_y.to_numpy().ravel()
        
        self.results_df = pd.DataFrame(columns=["Model", "Use LASSO", "PrecisionRecall"])
        
        self.path2figures = path2figures
        if not os.path.exists(self.path2figures):
            os.mkdir(self.path2figures)
            
    def lasso_feature_selection(self, n_cv):
        lasso = LassoCV(cv = n_cv, tol=1e-4, random_state = self.random_state)
        lasso.fit(self.train_x, self.train_y)
        self.selected_features = self.train_x.columns[lasso.coef_ != 0]
        
        self.train_x_reduced = self.train_x[self.selected_features]
        val_x_reduced = self.val_x[self.selected_features]
        
        return self.train_x_reduced, val_x_reduced
   
    def plot_results(self, y_pred, n, use_lasso, model_type):
        if use_lasso:
            lasso = 'Lasso_reduced'
        else:
            lasso = '(no_Lasso)'
        
        cm = confusion_matrix(self.val_y, np.where(y_pred > 0.5, 1, 0), labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fraud', 'NoFraud'])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix for {model_type} {lasso}")
        plt.show()
        
        precision, recall, thresholds = precision_recall_curve(self.val_y, y_pred)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if use_lasso:
            plt.title(f"n_trees = {n} Precision-recall curve with {self.train_x_reduced.shape[1]} variables")
        else:
            plt.title("Precision-recall curve")
        plt.legend()
        plt.grid()
        plt.show()
        return pr_auc
        
    def update_error(self, model_name, use_lasso, rmse):
        # As the results_df dataframe is not going to be too big
        # we can simply grow it
        self.results_df.loc[len(self.results_df)] = model_name, use_lasso, rmse
    
    def selected_model(self, n, model_type):
        if model_type == 'RF':
            md = RandomForestClassifier(n_estimators=n, random_state=self.random_state)
        elif model_type == 'SVM':
            md = SVC(probability=True)
        elif model_type == 'XGB':
            md = XGBClassifier(
                n_estimators=n, 
                max_depth=20, 
                learning_rate=0.001, 
                objective='binary:logistic')
        elif model_type == "linear":
            md = LogisticRegression()
        elif model_type == "ridge":
            md = LogisticRegression(penalty='l2', solver='saga', random_state=self.random_state)
        elif model_type == "elastic_net":
            md = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=self.random_state)

        return md
    
    def fit_model(self, model_type, n = None, use_lasso = False):
        
        if use_lasso:
            train_x, val_x = self.lasso_feature_selection(self.n_cv)
            
        else:
            train_x = self.train_x
            val_x = self.val_x
        
        mean_scores = []
        if isinstance(n, list):
            trained_models = []
            for nn in n:
                md = self.selected_model(nn, model_type)
                md.fit(train_x, self.train_y)
                y_probs = md.predict_proba(val_x)[:, 1]
                sco = self.plot_results(y_probs, nn, use_lasso, model_type)
                mean_scores.append(sco)
                trained_models.append(md)
                                   
            n_sco = float('inf') 
            self.n_tree = None  
            for i, (sco, tree) in enumerate(zip(mean_scores, n)):
                if sco < n_sco:  
                    pr = sco
                    self.n_tree = tree
                    model2restore = trained_models[i]
            
        elif isinstance(n, (float, int)) or not n:
            
            if n:
                print(f'Using {int(n)}')
                n = int(n)
                
            md = self.selected_model(n, model_type)
            md.fit(train_x, self.train_y)

            y_pred = md.predict_proba(val_x)[:, 1]  

            pr = self.plot_results(y_pred, n, use_lasso, model_type)
        
            model2restore = md
        
        else:
            ValueError("Unsupported format: {n}. Change it to list, int or None")
        
        self.update_error(model_type, use_lasso, pr)
        print(f" {model_type} Precision-Recall AUC: {pr:.3f}, lasso: {use_lasso}")
        
        return model2restore, pr
