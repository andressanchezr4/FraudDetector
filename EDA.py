# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andres.sanchez
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import time

class EDA(object):
    def __init__(self, data, path2figures):
        self.data = data
        self.path2figures = path2figures
        
        if not os.path.exists(self.path2figures):
            os.mkdir(self.path2figures)
          
        self.nan_figures = self.path2figures + 'nan_figures/'
        if not os.path.exists(self.nan_figures):
            os.mkdir(self.nan_figures)
        
        self.categorical_figures = self.path2figures + 'categorical_figures/'
        if not os.path.exists(self.categorical_figures):
            os.mkdir(self.categorical_figures)
            
        self.numerical_figures = self.path2figures + 'numerical_figures/'
        if not os.path.exists(self.numerical_figures):
            os.mkdir(self.numerical_figures)
        
    def general_overview(self):
        
        print("Dataset Info:")
        self.data.info()
        
        time.sleep(3)
        print('\nVariable dtypes:')
        print(self.data.dtypes.value_counts())

        time.sleep(3)
        print("\nMissing Values:")
        missing_values = self.data.isna().sum()
        print(missing_values[missing_values > 0])
        
        print('\nDescriptive Statistics (Numerical):')
        time.sleep(3)
        print(self.data.describe(include=[np.number]))

        print('\nDescriptive Statistics (Categorical):')
        time.sleep(3)
        print(self.data.describe(include=['object', 'category']))
    
    
    def NaN_analysis(self, missing_value_threshold = 'dummy', missing_prop = 'dummy'):
        
        # Heatmap of missing values
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.savefig(self.nan_figures + "missing_values_heatmap.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()

        missing_values = self.data.isna().sum()
        columns_with_missing = missing_values[missing_values > 0]

        # Histogram of missing values
        plt.figure(figsize=(10, 6))
        bars = columns_with_missing.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')

        for bar in bars.containers[0]:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", 
                     ha='center', va='bottom', fontsize=10, color='black')

        plt.title("Count of Missing Values Per Column")
        plt.ylabel("Number of Missing Values")
        plt.xlabel("Columns")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.nan_figures + "missing_values_barplot.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()

        missing_value_threshold1 = columns_with_missing.sort_values().quantile(0.7)
        missing_value_threshold2 = 0.3 * len(self.data) 
        
        self.columns2delete = columns_with_missing[(columns_with_missing > missing_value_threshold1) |
                                         (columns_with_missing > missing_value_threshold2)].index
        
        print(f'It may be a good idea to delete: {self.columns2delete.tolist()}')
    
    def repare_nan(self, nan_columns = []):
        
        if not nan_columns:
            nan_columns = self.columns2delete
        
        self.data = self.data.drop(nan_columns, axis=1)
        
        print(f'Columns deleted: {self.columns2delete}')
        
        return self.data
    
    def visualize_data(self):
        
        print(f'\nSaving the Distribution of Numerical Variables to {self.numerical_figures}')
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_cols].hist(figsize=(15, 10), bins=20, color='blue', edgecolor='black')
        plt.suptitle("Histograms of Numerical Variables")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.numerical_figures + "numerical_variables_hist.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()
        
        print(f'\nSaving the Distribution of Categorical Variables to {self.categorical_figures}')
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=self.data[col], order=self.data[col].value_counts().index, palette="viridis")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.categorical_figures + f"{col}_categories_distribution.png", 
                        dpi=300, 
                        bbox_inches='tight')
            plt.show()

        print(f'\nSaving Correlation Heatmap to {self.numerical_figures}')
        plt.figure(figsize=(12, 8))
        corr_matrix = self.data[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                    fmt=".2f", vmin=-1, vmax=1,
                    annot_kws={"size": 6})
        plt.title("Correlation Heatmap of Numerical Variables")
        plt.tight_layout()
        plt.savefig(self.numerical_figures + "numerical_correlation_heatmap.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()

        # Identify columns with correlation above 0.9
        correlation_threshold = 0.9
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.high_correlation = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    
    def corr_repare(self, correlated_columns = []):
        
        if not correlated_columns:
            correlated_columns = self.high_correlation
        
        self.data = self.data.drop(correlated_columns, axis=1)
        
        print(f'Columns deleted: {correlated_columns}')
                      
def one_hot_encode(df, columns):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns), index=df.index)
    return pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)

def label_encode(df, columns, mappings):
    if len(columns) != len(mappings):
        raise ValueError("The number of columns and mappings must match.")

    encoder = OrdinalEncoder(categories=mappings, handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[columns] = encoder.fit_transform(df[columns])
    
    return df

def impute_data(df, predicted_column, columns2ohc, columns2order, 
                predicted_variable, n_neighbors=5, joined = False,
                size_split = -1):
    
    # Dummy filling the nan so we can actually perform imputation
    temp_filler = SimpleImputer(strategy='constant', fill_value='missing')
    df[columns2ohc + columns2order] = temp_filler.fit_transform(df[columns2ohc + columns2order])

    df_encoded = one_hot_encode(df, columns2ohc)

    ordinal_mappings = [df[col].value_counts().index.tolist() for col in columns2order]
    df_encoded = label_encode(df_encoded, columns2order, ordinal_mappings)

    if joined:
        
        df_train2impute = df_encoded[:size_split]
        df_train2impute['isFraud'] = predicted_variable
        num_imputer = SimpleImputer(strategy='mean')
        df_imputed_train = num_imputer.fit_transform(df_train2impute)
        df_imputed_train = pd.DataFrame(df_imputed_train, columns=df_train2impute.columns, index=df_train2impute.index)
        
        df_test2impute = df_encoded[size_split:]
        num_imputer = SimpleImputer(strategy='mean')
        df_imputed_test = num_imputer.fit_transform(df_test2impute)
        df_imputed_test = pd.DataFrame(df_imputed_test, columns=df_test2impute.columns, index=df_test2impute.index)

        return df_imputed_train, df_imputed_test
        
    else:
        
        num_imputer = SimpleImputer(strategy='mean')
        imputed_data = num_imputer.fit_transform(df_encoded)
        df_imputed = pd.DataFrame(imputed_data, columns=df_encoded.columns, index=df.index)
        df_imputed ['isFraud'] = predicted_variable
        
        return df_imputed

    

    


    