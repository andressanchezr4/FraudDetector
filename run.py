# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andres.sanchez
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from DLs import Autoencoder, LossThresholdCallback, lr_schedule, plot_training_loss
from MLs2 import MLsPipeline
from Superlearner import SuperLearner
from EDA import EDA, impute_data

folder_path = './FraudDetector/'
path = folder_path + 'data/'

df_train1 = pd.read_csv(path + 'train_identity.csv')
df_train2 = pd.read_csv(path + 'train_transaction.csv')
df_train = df_train1.merge(df_train2, how = 'right', on = 'TransactionID')
df_train_noid = df_train.drop('TransactionID', axis=1) 

df_test1 = pd.read_csv(path + 'test_identity.csv')
df_test2 = pd.read_csv(path + 'test_transaction.csv')
df_test = df_test1.merge(df_test2, how = 'right', on = 'TransactionID')
df_test_noid = df_test.copy()

superlearner_oof_X = pd.read_csv(path + 'oof_X.csv')
superlearner_oof_y = pd.read_csv(path + 'oof_y.csv')


###########
### EDA ###
###########
# Threshold to drop rows with missing data
threshold_test = int(len(df_test_noid.columns) * 0.3) + 1
threshold_train = int(len(df_train_noid.columns) * 0.3) + 1

# Drop rows where the number of NaN values exceeds the threshold
df_train_noid = df_train_noid.dropna(thresh=len(df_train_noid.columns) - threshold_train, axis = 0)

# Explore and repare TRAIN DATA
path2figures = folder_path + 'figures/'
exploratory_analysis = EDA(df_train_noid, path2figures)

# exploratory_analysis.general_overview()
# exploratory_analysis.visualize_data()

# exploratory_analysis.high_correlation
# exploratory_analysis.corr_repare()

exploratory_analysis.NaN_analysis()
exploratory_analysis.columns2delete
df_train_nonan = exploratory_analysis.repare_nan()

# Explore and repare TEST DATA
path2figures = folder_path + 'figures_test/'
exploratory_analysis_test = EDA(df_test_noid, path2figures)

# exploratory_analysis_test.general_overview()
# exploratory_analysis_test.visualize_data()

# exploratory_analysis_test.high_correlation
# exploratory_analysis_test.corr_repare()

exploratory_analysis_test.NaN_analysis()
exploratory_analysis_test.columns2delete
df_test_nonan = exploratory_analysis_test.repare_nan()

df_test_ids2send = df_test_nonan.TransactionID # saved for later

########################
### Data Preparation ###
########################
# Data is joined for proper category level vectorization
common_columns = df_train_nonan.columns.intersection(df_test_nonan.columns)

# The sale price is saved to be added to the train set for proper imputation
df_train_isfraud = df_train_nonan.isFraud

df_test_nonan = df_test_nonan[common_columns]
df_train_nonan = df_train_nonan[common_columns]

size_train = len(df_train_nonan)
df2impute = pd.concat([df_train_nonan, df_test_nonan])

nominal_columns = df_train_nonan.describe(include=['object', 'category']).columns.tolist()
ordinal_columns = []

# DISCLAIMER: Imputation does not take information from test and train
# together to impute data, as that would corrupt the training data
# (data leakage) making it easier for the model to make predictions on the test set.
# We only join them to simplify the process of vectorising the categorical variables.
df_train_imputed, df_test_imputed = impute_data(df2impute, 
                                                'isFraud',
                                                nominal_columns, 
                                                ordinal_columns, 
                                                df_train_isfraud.tolist(), 
                                                n_neighbors=5, 
                                                joined = True, 
                                                size_split = size_train)

# TRAIN DATA preparation
df_train_imp, df_val_imp = train_test_split(df_train_imputed, 
                                            test_size=0.1,  
                                            stratify=df_train_imputed['isFraud'], random_state=1234)

train_y, train_x = df_train_imp[['isFraud']], df_train_imp.drop(['isFraud'], axis=1) 
val_y, val_x = df_val_imp[['isFraud']], df_val_imp.drop(['isFraud'], axis=1)

###############################
### Machine learning Models ###
###############################
path2scatterplots = folder_path + 'ml_predictions/'
pipeline = MLsPipeline(train_x, train_y, val_x, val_y, path2scatterplots)

# ---- Linear models ----
# Linear regression
lm_model, pr = pipeline.fit_model('linear')
lm_lasso, pr_lasso = pipeline.fit_model(model_type="linear", use_lasso=True)

# Ridge regression
lm_ridge, pr_ridge = pipeline.fit_model('ridge')
lm_ridge, pr_ridge = pipeline.fit_model('ridge', use_lasso=True)

# Elastic Net regression
lm_ridge, pr_ridge = pipeline.fit_model('elastic_net')
lm_ridge, pr_ridge = pipeline.fit_model('elastic_net', use_lasso=True)

# ---- Random Forrest ----
n_trees = [10, 50, 100, 200, 300, 500, 1000] # it also accepts ints
best_rf_model, best_rf_pr = pipeline.fit_model('RF', n_trees)
best_rf_model, best_rf_pr = pipeline.fit_model('RF', 50, use_lasso = True)

# ----- SVM -----
best_rf_model, best_rf_pr = pipeline.fit_model('SVM')
best_rf_model, best_rf_pr = pipeline.fit_model('SVM', use_lasso=True)

# ----- XGBoost ----- 
n_estimators = [100, 300, 500, 700, 900, 1200] # it also accepts ints
best_rf_model, best_rf_pr = pipeline.fit_model('XGB', n_estimators)
best_rf_model, best_rf_pr = pipeline.fit_model('XGB', 50, use_lasso = True)

print(pipeline.results_df)
# pipeline.results_df.to_csv(path2scatterplots +'results_mls.csv', index = None)

################################
### Stacked ML: SuperLearner ###
################################
my_sl = SuperLearner(train_x, train_y)
# my_sl.fit()

my_sl.oof_X = superlearner_oof_X.to_numpy()
my_sl.oof_y = superlearner_oof_y.to_numpy()
my_sl.alreadytrainedoof()

y_predproba_val = my_sl.predict(val_x)
precision, recall, _ = precision_recall_curve(val_y, y_predproba_val)
pr = auc(recall, precision)
print(f'SuperLearner PR: {pr:.3f}')
my_sl.evaluate_models(val_x, val_y) 

#######################
### DL: Autoencoder ###
#######################

# AUTOENCODER FOR ANOMALY DETECTION
scaler = StandardScaler()

# Standardize the dataset and train only with non fraud rows
train_x_ok = train_x[train_y.isFraud == 0.0]
val_x_ok = val_x[val_y.isFraud == 0.0]

standardized_data = scaler.fit_transform(train_x_ok)
standardized_data_val = scaler.fit_transform(val_x_ok)

autoencoder = Autoencoder(standardized_data)

# Not the best option to stop training but for the shake of the example
loss_threshold_callback = LossThresholdCallback(threshold=0.1)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
autoencoder.compile(optimizer=optimizer)

# Train the autoencoder
history = autoencoder.fit(standardized_data, standardized_data, epochs=100, 
                          batch_size=512, validation_data=(standardized_data_val, standardized_data_val),
                          callbacks=loss_threshold_callback
                          )
plot_training_loss(history.history)
autoencoder.plot_latent_space(standardized_data, train_y)

conf_mat = autoencoder.AE4AnomDetect(scaler.fit_transform(val_x), val_y)

# AUTOENCODER FOR DIMENSIONAL SPACE REDUCTION
standardized_data = scaler.fit_transform(train_x)
standardized_data_val = scaler.fit_transform(val_x)

autoencoder = Autoencoder(standardized_data)

loss_threshold_callback = LossThresholdCallback(threshold=0.1)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
autoencoder.compile(optimizer=optimizer)

history = autoencoder.fit(standardized_data, standardized_data, epochs=100, 
                          batch_size=512, validation_data=(standardized_data_val, standardized_data_val),
                          callbacks=loss_threshold_callback
                          )

decoded_data, latent_space = autoencoder.predict(standardized_data)
decoded_data_val, latent_space_val = autoencoder.predict(standardized_data_val)

pipeline = MLsPipeline(latent_space, train_y, latent_space_val, val_y, path2scatterplots)
xgb_model, PR_xgb = pipeline.fit_xgboost(500)
print(f"XGBoost PR: {PR_xgb:.3f}") 

# Test data
# standardized_data_test = scaler.fit_transform(df_test_imputed)
# decoded_data_test, latent_space_test = autoencoder.predict(standardized_data_test)

##################
### Submission ###
##################
# We train the best model with the whole train data and make predictions 
whole_train_x = df_train_imputed.drop(['isFraud'], axis = 1)
whole_train_y = df_train_imputed[['isFraud']]

xgb_model.fit(whole_train_x, whole_train_y)
y_pred_test = xgb_model.predict_proba(df_test_imputed.to_numpy())[:, 1]

df2send = pd.DataFrame([df_test_ids2send, y_pred_test.flatten()]).transpose()
df2send.columns = ['TransactionID', 'isFraud']
df2send['TransactionID'] = df2send['TransactionID'].astype('int64')

df2send.to_csv('./submission_xgb.csv', index = None) 




