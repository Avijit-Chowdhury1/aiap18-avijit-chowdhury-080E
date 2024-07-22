# Import needed libraries
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import requests
import math
import sqlite3 as sql
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score



#some color for use
GREEN = '\033[92m'
RESET = '\033[0m'

#LOAD Data

# Function to query the SQLite database
def query_database(path, query):
    connection = sql.connect(path) 
    df= pd.read_sql_query(query, connection)
    connection.close()
    return df

# Query downloaded databases
weather_data = query_database('weather.db',"SELECT * FROM weather")
air_data = query_database('air_quality.db',"SELECT * FROM air_quality")

## Based on exploratory data analysis, preprocess data

#merge the two datasets based on data_ref and date
merged_data = pd.merge(air_data, weather_data,on=['data_ref','date'], how='inner')

if 'data_ref' in merged_data.columns:
    merged_data.drop(columns=['data_ref'], inplace=True)

# Replace '--' and '-' with NaN
merged_data.replace('--', np.nan, inplace=True)
merged_data.replace('-', np.nan, inplace=True)

#Convert these variables to numeric
# Define a function to convert a column to numeric
def convert_to_numeric(column):
    # Replace '--' and other non-numeric placeholders with NaN
    column = column.replace('--', np.nan)
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(column)

# List of variables to convert (based on looking at the data and also the description given in the assignment sheet)
columns_to_convert = ['pm25_north', 'pm25_south', 'pm25_east','pm25_west', 'pm25_central', 'psi_north', \
 'psi_south', 'psi_east', 'psi_west', 'psi_central', 'Daily Rainfall Total (mm)','Highest 30 Min Rainfall (mm)', \
  'Highest 60 Min Rainfall (mm)','Highest 120 Min Rainfall (mm)', 'Min Temperature (deg C)','Maximum Temperature (deg C)', \
   'Max Wind Speed (km/h)','Min Wind Speed (km/h)']

# Loop through the columns and apply the conversion function
for col in columns_to_convert:
    merged_data[col] = convert_to_numeric(merged_data[col])


# Select only the numerical columns
numerical_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns

# Apply KNN imputer for missing values
knn_imputer = KNNImputer(n_neighbors=5)
merged_data[numerical_cols] = knn_imputer.fit_transform(merged_data[numerical_cols])
merged_data_nomissing=merged_data

# Make all negative values in the column "Max Wind Speed (km/h)" positive
merged_data_nomissing['Max Wind Speed (km/h)'] = merged_data_nomissing['Max Wind Speed (km/h)'].abs()

rain_columns = [ 'Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)' ]

# Create a categorical variable for rain presence with 'Yes' or 'No'
merged_data_nomissing['Rain_Presence'] = np.where(merged_data_nomissing[rain_columns].sum(axis=1) > 0, 'Yes', 'No')

# Create a composite variable by averaging the PM2.5 variables
merged_data_nomissing['Average_PM25'] = merged_data_nomissing[['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']].mean(axis=1)
merged_data_nomissing['Average_PSI'] = merged_data_nomissing[['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']].mean(axis=1)

pm25_columns_to_drop = ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']
psi_columns_to_drop = ['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']
rain_columns_to_drop = ['Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)']

columns_to_drop = pm25_columns_to_drop + psi_columns_to_drop + rain_columns_to_drop

# Drop the columns if they exist in the DataFrame
for col in columns_to_drop:
    if col in merged_data_nomissing.columns:
        merged_data_nomissing.drop(columns=[col], inplace=True)


def remove_outliers_iqr(df):

    # Select only the numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    original_row_count = df.shape[0]
    
    # Dictionary to store percentage of data removed for each column
    removed_data_percentage = {}
    
    # Remove outliers for each numerical column
    for column in numerical_cols:
        original_col_count = df[column].notna().sum()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        new_col_count = df[column].notna().sum()
        removed_data_percentage[column] = ((original_col_count - new_col_count) / original_col_count) * 100
    
    # Print the percentage of data removed for each variable
    for column, percentage in removed_data_percentage.items():
        print(f"Percentage of data removed from {column}: {percentage:.2f}%")
    
    return df

merged_data_nooutliers=remove_outliers_iqr(merged_data_nomissing)

if 'Relative Humidity (%)' in merged_data_nooutliers.columns:
    # we can bifurcate and create a binary variable for relative humidity saturation
    merged_data_nooutliers['RH_Saturation'] = np.where(merged_data_nooutliers['Relative Humidity (%)'] >= 100, 'Yes', 'No')

# now delete the original variable
if 'Relative Humidity (%)' in merged_data_nooutliers.columns:
     merged_data_nooutliers.drop(columns=['Relative Humidity (%)'], inplace=True)

# Feature engineer month, year, day
if 'date' in merged_data_nooutliers.columns:
    merged_data_nooutliers['date'] = pd.to_datetime(merged_data_nooutliers['date'], format='%d/%m/%Y')

    # Extract features from the 'date' column
    merged_data_nooutliers['month'] = merged_data_nooutliers['date'].dt.month
    merged_data_nooutliers['day_of_week'] = merged_data_nooutliers['date'].dt.day_name()
    merged_data_nooutliers['year'] = merged_data_nooutliers['date'].dt.year
    merged_data_nooutliers = merged_data_nooutliers.drop(columns=["date"])

# Convert all values to lowercase
merged_data_nooutliers['Dew Point Category'] = merged_data_nooutliers['Dew Point Category'].str.lower()
merged_data_nooutliers['Wind Direction'] = merged_data_nooutliers['Wind Direction'].str.lower()

# Define a mapping dictionary

mapping1 = {
    'very high': 'VH',
    'vh': 'VH',
    'high': 'H',
    'high level': 'H',
    'h': 'H',
    'moderate': 'M',
    'm': 'M',
    'low': 'L',
    'l': 'L',
    'vl': 'VL',
    'very low': 'VL',
    'minimal': 'L',
    'below average': 'L',
    'extreme': 'VH',
    'normal': 'M' }


# Replace values using the mapping
merged_data_nooutliers['Dew Point Category'] = merged_data_nooutliers['Dew Point Category'].map(mapping1).fillna(merged_data_nooutliers['Dew Point Category'])

#Do the same for Wind Direction
mapping2 = {
    'w': 'West',
    'west': 'West',
    'w.': 'West',
    'nw': 'Northwest',
    'northwest': 'Northwest',
    'nw.': 'Northwest',
    'n': 'North',
    'northward': 'North',
    'north': 'North',
    'n.': 'North',
    'northeast': 'Northeast',
    'ne': 'Northeast',
    'northeast': 'Northeast',
    'ne.': 'Northeast',
    'se': 'Southeast',
    'southeast': 'Southeast',
    'se.': 'Southeast',
    's': 'South',
    'southward': 'South',
    'south': 'South',
    's.': 'South',
    'e': 'East',
    'east': 'East',
    'e.': 'East',
    'e': 'East',
    'southeast': 'Southeast',
    'southwest': 'Southwest',
    'sw': 'Southwest',
    'sw.': 'Southwest' }

# Replace values using the mapping
merged_data_nooutliers['Wind Direction'] = merged_data_nooutliers['Wind Direction'].map(mapping2).fillna(merged_data_nooutliers['Wind Direction'])

# check duplicates and remove 
print("Number of duplicates:", merged_data_nooutliers.duplicated().sum())

merged_data_nooutliers = merged_data_nooutliers[~merged_data_nooutliers.duplicated()]
print("Removed duplicates. Remaining duplicates:", merged_data_nooutliers.duplicated().sum())

#combine categories where levels are sparse
def combine_rare_categories(df, col, threshold=0.05):
    """Combine rare categories in a column."""
    value_counts = df[col].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    df[col] = df[col].apply(lambda x: 'M_below' if x in rare_categories else x)
    return df

# Combine rare categories in "Dew Point Category" 
merged_data_nooutliers = combine_rare_categories(merged_data_nooutliers, 'Dew Point Category', threshold=0.05)

## Define a function to run SMOTENC to upsample the target variable
def apply_SMOTENC(df,target_col):
    
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # remove the target column from in cat_features
    if target_col in cat_features:
        cat_features.remove(target_col)

    # get categorical feature indices
    categorical_feature_indices = [df.columns.get_loc(col) for col in cat_features]
 
    # Define the features and target for SMOTENC
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Recalculate categorical feature indices after splitting
    categorical_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]

    # Initialize SMOTENC
    smote_nc = SMOTENC(categorical_features=categorical_feature_indices, random_state=42)

    # Apply SMOTENC to the training data
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

    # Combine the resampled features and target into a single dataframe
    resampled_df = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns),
            pd.Series(y_train_resampled, name=target_col)], axis=1)
    
    return resampled_df

target_col = 'Daily Solar Panel Efficiency'

merged_upsampled = apply_SMOTENC(merged_data_nooutliers,target_col)

print("The script completed preprocessing, Now fitting models")

"""

Let's proceed with fitting three suitable machine learning models to predict the solar panel efficiency as.

We will use the following models:

Logistic Regression: A simple linear model that can be effective for multi-class classification.
Random Forest Classifier: An ensemble model that can handle complex interactions and is robust to overfitting.
Support Vector Machine (SVM): Support Vector Machine (SVM) is a powerful and versatile machine learning algorithm used for classification and regression tasks.

"""

df = merged_upsampled
# Define the features and target
X = df.drop(columns=['Daily Solar Panel Efficiency'])
y = df['Daily Solar Panel Efficiency']

# Identify categorical and numerical features
categorical_features_ohe = ['Wind Direction', 'Rain_Presence', 'RH_Saturation', 'Dew Point Category']  # List of features for OneHotEncoder
categorical_features_le = ['day_of_week','month','year']  # List of features for LabelEncoder
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Apply Label Encoding to the specified features
le = LabelEncoder()
for col in categorical_features_le:
    X[col] = le.fit_transform(X[col])

# Define the preprocessor for OneHotEncoder and StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat_ohe', OneHotEncoder(), categorical_features_ohe)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Combine transformed features with label encoded features
X_train_combined = np.hstack((X_train_transformed, X_train[categorical_features_le].values.reshape(-1, len(categorical_features_le))))
X_test_combined = np.hstack((X_test_transformed, X_test[categorical_features_le].values.reshape(-1, len(categorical_features_le))))

# Helper function to evaluate models
def evaluate_model(model, X_test, y_test, y_pred):
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Train and evaluate Logistic Regression model
print("Logistic Regression:")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_combined, y_train)
y_pred_log_reg = log_reg.predict(X_test_combined)
evaluate_model(log_reg, X_test_combined, y_test, y_pred_log_reg)

# Train and evaluate Random Forest model
print("\nRandom Forest:")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_combined, y_train)
y_pred_rf = rf_clf.predict(X_test_combined)
evaluate_model(rf_clf, X_test_combined, y_test, y_pred_rf)

# Train and evaluate SVM model
print("\nSVM:")
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train_combined, y_train)
y_pred_svm = svm_clf.predict(X_test_combined)
evaluate_model(svm_clf, X_test_combined, y_test, y_pred_svm)

# Feature Importances from Random Forest
importances = rf_clf.feature_importances_
encoded_feature_names = numerical_features + list(preprocessor.named_transformers_['cat_ohe'].get_feature_names_out(categorical_features_ohe)) + categorical_features_le
feature_importances = pd.DataFrame({'feature': encoded_feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
