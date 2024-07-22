Introduction
The project aims to employ Machine Learning (ML) models to effectively predict Daily Solar Panel Efficiency levels (High, Medium, Low) from daily weather and air_quality data. The analysis includes data cleaning, outlier detection and handling, correlation analysis, and feature engineering. Key findings and recommendations are highlighted to enhance the predictive power of future models. Three different ML models are tested, and their performance is reported.

Preprocessing steps

1.	‘air_quality’ and ‘weather’ data were first merged using the common unique identifies between these two datasets. Merging was conducted so that all available features could be used simultaneously to build machine learning models. *weather* and *air_quality* data have an unequal number of rows and features. The available features in these datasets also do not overlap. *air_quality* has more datapoints than *weather*. The key to merging these two dataframes is the unique identifier variable *data_ref* and *date*.

2.	Tdata types for each column were examined to see if they were accurate. Accurate specification of data type is important to ensure they are encoded accordingly in any Machine learning models. Many of the variables which are non-categorical (example: Daily Rainfall Total (mm), Highest 30 Min Rainfall (mm), etc.) are saved as *object* variables. This was found to be due to the presence of non-numerical text in these variables, which were converted to NaN to assign the variable a numeric datatype.

3.	Some of the variables had missing values (i.e., total data point is lower than the total number of rows). Identifying and handling missing values is crucial to ensure data quality. The percentage of missing values for some of the numerical values were ~ 10% - we would lose substantial data if we were to discard them. Replacing these values with mean/median may also not be suitable given the number of outliers. The missing values were filled with imputed values using a K-nearest neighbor imputer.  Categorical variables have no missing values.

4.	The distributions of numeric variables were first plotted. Skewed Distributions. Many variables such as PM2.5 levels (pm25_north, pm25_south, pm25_east, pm25_west, pm25_central), PSI levels, rainfall metrics, Sunshine Duration, Relative Humidity (%), and Cloud Cover show skewed distributions. This indicates that most observations are clustered towards the lower end of the range, with a few high-value outliers. Many machine-learning and statistical models perform robustly only when normality assumptions are met in the data. Removing outliers (see the subsequent point on outlier removal) rendered some of these variables to conform to normality. 

5.	The rainfall metrics ['Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)'] have a significant amount of zero-valued data (i.e., absence of rain). A general 'Rain_Presence' (i.e., Rainfall > 0mm) variable was constrcuetd to simplify the analysis.

6.	Min Temperature (deg C), Max Temperature (deg C), Min Wind Speed (km/h), and Air Pressure had more symmetric distributions, suggesting more evenly distributed data around the mean. Max Wind Speed (km/h) has some negative values which are likely data-entry errors. These were replaced with the absolute values. Wet Bulb Temperature (deg F) has a few negative values and most values are positive, making the variable non-normal.

7.	Correlation between numeric variables: Some of the variables may also carry very similar information (highly correlated), and if so, can introduce multicollinearity in our analysis. Multicollinearity is an issue in our analysis because it leads to redundancy among features, making it difficult to determine the individual effect of each predictor on the target variable. This can result in unstable estimates of regression coefficients, inflated standard errors, and reduced model interpretability, ultimately compromising the reliability and accuracy of the model. To avoid this, new composite variables were created from highly correlated variables by taking their average.

8.	Outlier removal is important in building machine learning models because outliers can distort and skew the training data, leading to poor model performance and inaccurate predictions. By removing outliers, the model can learn the true underlying patterns of the data, resulting in improved accuracy and robustness. The Inter Quartile Range (IQR) method is effective for outlier removal because it is based on the middle 50% of the data, making it less sensitive to extreme values and applicable to various data distributions. Additionally, IQR is simple to calculate and understand, providing a straightforward way to enhance the quality of the dataset used for model training. IQR based outlier removal was used to remove outliers from each feature. The percentage of data removed did not exceed 10% for any of the variables:

Percentage of data removed from Min Temperature (deg C): 0.94%
Percentage of data removed from Maximum Temperature (deg C): 2.75%
Percentage of data removed from Min Wind Speed (km/h): 0.54%
Percentage of data removed from Max Wind Speed (km/h): 2.84%
Percentage of data removed from Sunshine Duration (hrs): 7.80%
Percentage of data removed from Cloud Cover (%): 6.27%
Percentage of data removed from Wet Bulb Temperature (deg F): 6.13%
Percentage of data removed from Relative Humidity (%): 9.61%
Percentage of data removed from Air Pressure (hPa): 0.24%
Percentage of data removed from Average_PM25: 5.86%
Percentage of data removed from Average_PSI: 0.26%

9.	All variables except *Relative Humidity* resembled a normal distribution following the previous data cleaning and feature engineering steps. Applying a log transformation did not remedy this skewness. The variable was bifurcated and a binary variable (Yes, No) for relative humidity saturation was created.

10.	Feature engineering: Feature engineering was performed to get the month, year, and day of the week from the date. The month, year, and day of the week may be more informative (e.g. seasonal information) in predicting the outcome than the date, which is an arbitrary value.

11.	In the categorical variables *Dew Point Category* and *Wind Direction*, the values are not consistently indicative of specific categories. For example, "Very High" is input as "VH" or "very high" in *Dew Point Category*. Similarly, 'NW' is input as 'northwest', 'NORTHWEST', etc. To ensure the correct number of unique values (i.e., levels) for each categorical variable, these entries were made consistent.

12.	Any duplicates in the data were removed as they are redundant and can lead to overfitting.

13.	Looking at the frequency distribution of categorical features helped identify any unbalanced features, which can hamper model performance. The 'Dew Point Category' has an uneven distribution with 'H' (High) being the most frequent category, followed by 'VH' (Very High), and 'L' (Low). 'M' (Medium) and 'VL' (Very Low) are the least frequent. As the frequency of all levels except 'VH' and 'H' is very rare, these were combined to a single category 'M_below'. The presence of unbalanced data also prompts the consideration of choosing ML prediction models that are more robust to imbalanced datasets (such as Decision Trees and Random Forests) or that can adjust the class weights (such as SVM and Logistic Regression).



14.	The target variable 'Daily Solar Panel Efficiency' shows a skewed distribution with 'Medium' being the most frequent, followed by 'High' and 'Low'. Upsampling to increase the number of samples in the minority classes (low, high) using (SMOTENC: Synthetic Minority Over-sampling Technique for Nominal and Continuous) remedied this. This process did not significantly change the distribution of other variables.

15.	To examine the relation between the target and features, visual inspection of the distribution of the numerical variables as a function of Daily Solar Panel Efficiency was conducted. Looking at the target variable as a function of the various numeric variables, *Sunshine hours* seems to positively predict HIGH efficiency, while *cloud cover* predicts LOW efficiency. Maximum wind speed also seems to lower the efficiency. An apparent observation was that Efficiency is lower on days with Rainfall versus without.

16.	Handling Categorical Variables: Both OneHotEncoder and LabelEncoder were employed to handle categorical variables appropriately. This ensured that the categorical features were transformed into a format suitable for machine learning models without losing important information.

17.	Feature Importance Analysis: A Random Forest classifier was trained on the dataset to determine the importance of each feature. The feature importance analysis revealed that Sunshine duration, Max/Min Wind Speed, and Cloud Cover have the most significant impact on the target variable, guiding future feature selection and engineering efforts.

 

Overview of key findings from the EDA and its implications for the ML pipeline: Data required significant cleaning (imputing missing values, outlier removal) and some variables that were highly correlated with each other were averaged into a single variable for better model fit. The target variable is unbalanced therefore oversampling using SMOTENC was conducted to balance the target variable counts. The model choice should also be driven by robustness to imbalanced datasets (such as Decision Trees and random Forests) or that can adjust the class weights (such as SVM and Logistic Regression). Other feature engineering included the creation of month, year, and day variables from ‘date’ which may be more representative of weather and seasonal fluctuations.

Explanation of Choice of Models

Based on the categorical nature of the target variable, as its inherently unbalanced distribution, the following models were chosen:

Linear Regression: For its simplicity and interpretability.
Random Forest: For its ability to handle non-linear relationships and feature importance scoring.
Support Vector Machine (SVM): For its effectiveness in high-dimensional spaces and robustness against overfitting.

Evaluation of models
The models were evaluated based on the following metrics:

Accuracy: Measures the overall correctness of the model.
Precision: Measures the proportion of true positive predictions among all positive predictions.
Recall: Measures the proportion of true positive predictions among all actual positives.
F1 Score: Harmonic mean of precision and recall, providing a balance between the two.

Model performance:

For Logistic Regression, the results were as follows: an Accuracy of 0.6679, Precision of 0.6683, Recall of 0.6679, and an F1 Score of 0.6670. The classification report showed precision, recall, and F1 scores for each class (High, Low, and Medium) as well as macro and weighted averages. The confusion matrix indicated that out of 275 instances of the High class, 204 were correctly predicted as High, 27 were incorrectly predicted as Low, and 44 were incorrectly predicted as Medium. Similarly, for the Low class, 181 were correctly predicted, 41 were incorrectly predicted as High, and 53 as Medium. For the Medium class, 166 were correctly predicted, 61 were incorrectly predicted as High, and 48 as Low.

For Random Forest, the results were significantly better with an Accuracy of 0.8497, Precision of 0.8536, Recall of 0.8497, and an F1 Score of 0.8503. The classification report indicated higher precision, recall, and F1 scores across all classes compared to Logistic Regression. The confusion matrix showed that for the High class, 228 instances were correctly predicted, 12 were incorrectly predicted as Low, and 35 as Medium. For the Low class, 233 were correctly predicted, 13 were incorrectly predicted as High, and 29 as Medium. For the Medium class, 240 were correctly predicted, 10 were incorrectly predicted as High, and 25 as Low.

For the Support Vector Machine (SVM), the results were: an Accuracy of 0.6848, Precision of 0.6863, Recall of 0.6848, and an F1 Score of 0.6848. The classification report showed balanced precision, recall, and F1 scores across all classes. The confusion matrix revealed that for the High class, 201 instances were correctly predicted, 28 were incorrectly predicted as Low, and 46 as Medium. For the Low class, 177 were correctly predicted, 39 were incorrectly predicted as High, and 59 as Medium. For the Medium class, 187 were correctly predicted, 44 were incorrectly predicted as High, and 44 as Low.

Confusion Matrix Analysis
The confusion matrices provide detailed insights into the model's performance:

Logistic Regression: Shows significant misclassifications, particularly between the High and Medium classes.
Random Forest: Has the highest accuracy and relatively fewer misclassifications, indicating it is the most effective model. For example, 240 out of 275 instances for the Medium class were correctly predicted.

SVM: Performs better than Logistic Regression but not as well as Random Forest, with moderate misclassifications across classes.

Feature Importance
The most important features identified by the Random Forest model are:

Sunshine Duration (hrs)
Cloud Cover (%)
Max Wind Speed (km/h)

These features contributed most significantly to the model's predictive performance, indicating their strong influence on the target variable.

Conclusion
Random Forest is the best-performing model based on the evaluation metrics, showing the highest accuracy, precision, recall, and F1 score. It also has the best performance in the confusion matrix analysis, indicating it is the most reliable for this dataset.







