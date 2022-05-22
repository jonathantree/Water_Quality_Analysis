# 2022 Data Analytics Final Project
## Identifying at-risk and underserved communities with poor quality drinking water in the United States

## Project Overview
The purpose of this project is to analyze drinking water quality to determine if the quality of the community's drinking water is correlated with certain demographic markers, such as income level. Specifically, we are aiming to use socioeconomic data and drinking water quality data to identify at-risk communities that are historically underserved. By identifying which communities are at most risk (i.e., poverty level, racial inequity) alongside our analysis of water quality data and our supervised machine learning model results, we can identify those high-priority communities. We can then have a subset of communities that can be targeted that need humanitarian support to remediate their water source.

### Why
Access to clean drinking water, free of chemicals and biological material, is a basic human right. In the United States, we take access to clean drinking water for granted. The lead in the drinking water in Flint, MI was a wake-up call that perhaps other communities are also dealing with sub-par water. 

### Question
Do communities with traditionally underserved demographics have access to clean drinking water?

### Team Communication 
We are utilizing the following channels for communication and co-working:
- Slack: for daily stand-ups and discussion or troubleshooting
- Microsoft OneNote: for project planning and note-taking
- Zoom: for co-working sessions

## Data
Data is being sourced from web scraping of [Environmental Working Group's Tap Water Database](https://www.ewg.org/tapwater/), and the US Census data (2010).

## Project Flow Outline

### Data Collection and Preprocessing for ML Feature Engineering
1. Census Data Workflow
   - Calculation of diversity indices
2. EWG Web scraping
   - The [`web scraping script`](/Web_Scraping/PyScripts/scrape_EWG.py) receives a state ID from the user. It then creates [directory](/Resources/Data/user_scrape_data/) for that state where all of the scraped data are stored. The script generates a [list](/Resources/Data/user_scrape_data/utilities.csv) of all of the water utilities in that state, visits each one, and pulls out the [data](/Resources/Data/user_scrape_data/contaminants.csv) contained on each site about contaminants. 
   - Cleaning of web scraped data is done by executing the [`clean_and_build_v2.py`](/Web_Scraping/PyScripts/clean_and_build_v2.py) which steps through each state directory created by the [`web scraping script`](/Web_Scraping/PyScripts/scrape_EWG.py) and generates the [cleaned data file](/Resources/Data/user_scrape_data/contaminants_cleaned.csv), and builds a master [dataset](Water_Quality_Analysis/Resources/Data/Cleaned_Data/) containing all of the scraped data to update the database.
   - To update the database, the [`update_DB.py`](/Database/update_DB.py) script is run, which then generates the final table that can be merged with the census data. This table is a county-level summary of the contamination data from web scraping.
3. Feature Engineering for ML 
   - To obtain the initial DataFrame for input into any of our ML models, the following SQLite query is performed which does an inner join on the census data table and the summary contaminants table.
      ``` python 
     df = pd.read_sql_query("SELECT * FROM Census_Data INNER JOIN Contaminant_Summary on Census_Data.county_FIPS = Contaminant_Summary.county_FIPS",conn)
     ```
 3. Target Engineering for Binary Classification
    - To have a target for ML models to predict an algorithm had to be developed to establish the weights of the features and determine a final priority level. In the case of the binary classification models, this target is a high-priority (1) or low-priority (0). The [algorithm](Water_Quality_Analysis/Priority_Algo_dev/Priority_algo_dev.ipynb) which establishes the binary target was used for all binary classification model development.
4. Supervised ML Binary Classification
   - The benefit of choosing a binary classifier model is that there are many models to choose from and many hyperparameters for tuning. The drawback is that we are limited to two target values (high-prority to low-priority). 
   - To gain access to predicting multiple levels of priorities, we are also exploring the development of an ordinal logistic regression model which will be developed in `R`. This model allows us to predict variables that are not only categorical but they are also following an order (low to high / high to low).
   - The target had a slight imbalance and some resampling techniques were explored but did not yield any better results.
   - The feature selection included the following columns:
      - pct_White                 float64
      -  pct_Black                 float64
      -  pct_Native                float64
      -  pct_Asian                 float64
      -  pct_Pacific_Islander      float64
      -  pct_Other                 float64
      -  pct_Not_White             float64
      -  pct_Hispanic              float64
      -  pct_Not_Hispanic          float64
      - pct_Two_or_more_Races     float64
      - Simpson_Race_DI           float64
      - Simpson_Ethnic_DI         float64
      - Shannon_Race_DI           float64
      - Shannon_Ethnic_DI         float64
      - Gini_Index                float64
      - Num_Contaminants            int64
      - Sum_ContaminantFactor       int64
      - Avg_Contaminant_Factor    float64
    - These features were selected due to their ability to characterize a county's racial distribution, income inequality, and, the overall water quality to the county residents have access. 
   - The top-performing model was XGBoost `GBClassifier` with a 98% accuracy
   - Binary Classification models test were:
      - Balanced Random Forest Classifier  
      - Easy Ensemble AdaBoost Classifier
      - GBClassifier
      - Other under/oversampling techniques were also explored but did not yield better results than the aforementioned models
- These models are summarized below in the following code:

```python
import sqlite3
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import train_test_split

```


```python
db = r'C:/Users/jonat/UO_Bootcamp/Group_project/git_Water_Quality_Analysis/Water_Quality_Analysis/Database/database.sqlite3'
# Connect to SQLite database
conn = sqlite3.connect(db)
  
# Create cursor object
cursor = conn.cursor()

#Read in the Data from the DB
df = pd.read_sql_query("SELECT * FROM Census_Data INNER JOIN Contaminant_Summary on Census_Data.county_FIPS = Contaminant_Summary.county_FIPS",conn)
```


```python
#Get the target binary data from the .csv file that was generated in the Priority_algo_dev.ipynb
target = pd.read_csv('data_with_binary_priority.csv', usecols=['Priority'])
```


```python
#The problem is the imbalanced data and this will need to be addressed
target.Priority.value_counts()
```




    0    616
    1    266
    Name: Priority, dtype: int64





```python
df.dtypes
```




    county_FIPS                 int64
    Geographic_Area_Name       object
    County                     object
    GEOID                      object
    Total_Population            int64
    White                       int64
    Black                       int64
    Native                      int64
    Asian                       int64
    Pacific_Islander            int64
    Other                       int64
    Two_or_more_Races           int64
    Hispanic                    int64
    Not_Hispanic                int64
    Not_White                   int64
    pct_White                 float64
    pct_Black                 float64
    pct_Native                float64
    pct_Asian                 float64
    pct_Pacific_Islander      float64
    pct_Other                 float64
    pct_Not_White             float64
    pct_Hispanic              float64
    pct_Not_Hispanic          float64
    pct_Two_or_more_Races     float64
    Simpson_Race_DI           float64
    Simpson_Ethnic_DI         float64
    Shannon_Race_DI           float64
    Shannon_Ethnic_DI         float64
    Gini_Index                float64
    County_FIPS                object
    Num_Contaminants            int64
    Sum_Population_Served       int64
    Sum_ContaminantFactor       int64
    Min_Contaminant_Factor     object
    Max_Contaminant_Factor     object
    Avg_Contaminant_Factor    float64
    dtype: object




```python
df.columns
```




    Index(['county_FIPS', 'Geographic_Area_Name', 'County', 'GEOID',
           'Total_Population', 'White', 'Black', 'Native', 'Asian',
           'Pacific_Islander', 'Other', 'Two_or_more_Races', 'Hispanic',
           'Not_Hispanic', 'Not_White', 'pct_White', 'pct_Black', 'pct_Native',
           'pct_Asian', 'pct_Pacific_Islander', 'pct_Other', 'pct_Not_White',
           'pct_Hispanic', 'pct_Not_Hispanic', 'pct_Two_or_more_Races',
           'Simpson_Race_DI', 'Simpson_Ethnic_DI', 'Shannon_Race_DI',
           'Shannon_Ethnic_DI', 'Gini_Index', 'County_FIPS', 'Num_Contaminants',
           'Sum_Population_Served', 'Sum_ContaminantFactor',
           'Min_Contaminant_Factor', 'Max_Contaminant_Factor',
           'Avg_Contaminant_Factor'],
          dtype='object')



## Feature Selection


```python
df_model = df.drop(columns=['county_FIPS', 
                            'Geographic_Area_Name', 
                            'County', 'GEOID',
                            'Total_Population',
                            'White', 
                            'Black', 
                            'Native', 
                            'Asian',
                            'Pacific_Islander', 
                            'Other', 
                            'Two_or_more_Races', 
                            'Hispanic',
                            'Not_Hispanic', 
                            'Not_White',
                            'County_FIPS',
                            'Sum_Population_Served',
                            'Min_Contaminant_Factor', 
                            'Max_Contaminant_Factor',
                            ])
```


```python
#Check to make sure the data types don't need fixing
df_model.dtypes
```




    pct_White                 float64
    pct_Black                 float64
    pct_Native                float64
    pct_Asian                 float64
    pct_Pacific_Islander      float64
    pct_Other                 float64
    pct_Not_White             float64
    pct_Hispanic              float64
    pct_Not_Hispanic          float64
    pct_Two_or_more_Races     float64
    Simpson_Race_DI           float64
    Simpson_Ethnic_DI         float64
    Shannon_Race_DI           float64
    Shannon_Ethnic_DI         float64
    Gini_Index                float64
    Num_Contaminants            int64
    Sum_ContaminantFactor       int64
    Avg_Contaminant_Factor    float64
    dtype: object




```python
#Check for Nan even though cleaning scripts should have excluded them by this stage
df_model.isna().sum()
```




    pct_White                 0
    pct_Black                 0
    pct_Native                0
    pct_Asian                 0
    pct_Pacific_Islander      0
    pct_Other                 0
    pct_Not_White             0
    pct_Hispanic              0
    pct_Not_Hispanic          0
    pct_Two_or_more_Races     0
    Simpson_Race_DI           0
    Simpson_Ethnic_DI         0
    Shannon_Race_DI           0
    Shannon_Ethnic_DI         0
    Gini_Index                0
    Num_Contaminants          0
    Sum_ContaminantFactor     0
    Avg_Contaminant_Factor    0
    dtype: int64



## Split the data into training and test data


```python
# Create our features
X = df_model
# Create our target
y = target
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
#Check the imbalance in the training set
y_train.value_counts()
```




    Priority
    0           460
    1           201
    dtype: int64



## Ensemble Learners

### Balanced Random Forest Classifier


```python
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
brf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1) 
brf_model.fit(X_train,y_train)
```




    BalancedRandomForestClassifier(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = brf_model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.9717948717948718




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>152</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>2</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.99      0.97      0.97      0.98      0.97      0.94       156
              1       0.94      0.97      0.97      0.95      0.97      0.94        65
    
    avg / total       0.97      0.97      0.97      0.97      0.97      0.94       221
    
    


```python
# List the features sorted in descending order by feature importance
sorted(zip(brf_model.feature_importances_, X.columns), reverse=True)
```




    [(0.21577329250084878, 'pct_Hispanic'),
     (0.18270331734030493, 'Simpson_Ethnic_DI'),
     (0.16933527790291714, 'Shannon_Ethnic_DI'),
     (0.14741969379439726, 'pct_Not_Hispanic'),
     (0.07618300294105276, 'pct_Other'),
     (0.06788945084338857, 'Shannon_Race_DI'),
     (0.04540258450048268, 'pct_Not_White'),
     (0.02858899205530442, 'Simpson_Race_DI'),
     (0.019062721712183664, 'pct_White'),
     (0.013419859566826287, 'pct_Asian'),
     (0.010463765501046053, 'pct_Two_or_more_Races'),
     (0.00537416069877027, 'pct_Black'),
     (0.005029362553335992, 'Sum_ContaminantFactor'),
     (0.003813001607423072, 'pct_Native'),
     (0.003033713296187055, 'Gini_Index'),
     (0.002441597706863405, 'Num_Contaminants'),
     (0.002387526716818439, 'Avg_Contaminant_Factor'),
     (0.0016786787618492632, 'pct_Pacific_Islander')]



### Easy Ensemble AdaBoost Classifier


```python
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier 
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train,y_train)
```




    EasyEnsembleClassifier(n_estimators=100, random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = eec.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.9685897435897436




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>151</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>2</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.99      0.97      0.97      0.98      0.97      0.94       156
              1       0.93      0.97      0.97      0.95      0.97      0.94        65
    
    avg / total       0.97      0.97      0.97      0.97      0.97      0.94       221
    
    

### Naive Random Oversampling


```python
# implement random oversampling
from imblearn.over_sampling import RandomOverSampler
# Resample the training data with the RandomOversampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

y_resampled.value_counts()
```




    Priority
    0           460
    1           460
    dtype: int64




```python
from sklearn.linear_model import LogisticRegression
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# make predictions
y_pred = model.predict(X_test)
```


```python
from sklearn.metrics import balanced_accuracy_score
#Calculate the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```




    0.8987179487179486




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>134</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>4</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.97      0.86      0.94      0.91      0.90      0.80       156
              1       0.73      0.94      0.86      0.82      0.90      0.81        65
    
    avg / total       0.90      0.88      0.92      0.89      0.90      0.80       221
    
    

### SMOTE Oversampling


```python
# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(
    X_train, y_train
)
y_resampled.value_counts()
```




    Priority
    0           460
    1           460
    dtype: int64




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.9019230769230769




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>135</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>4</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.97      0.87      0.94      0.92      0.90      0.81       156
              1       0.74      0.94      0.87      0.83      0.90      0.82        65
    
    avg / total       0.90      0.89      0.92      0.89      0.90      0.81       221
    
    

## Undersampling



```python
# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```




    Counter({'Priority': 1})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.5935897435897436




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>118</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>37</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.76      0.76      0.43      0.76      0.57      0.34       156
              1       0.42      0.43      0.76      0.43      0.57      0.32        65
    
    avg / total       0.66      0.66      0.53      0.66      0.57      0.33       221
    
    

## Combination (Over and Under) Sampling



```python
# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=1)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
y_resampled.value_counts()
```




    Priority
    1           192
    0           158
    dtype: int64




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)

```




    0.8865384615384615




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>135</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>6</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
              0       0.96      0.87      0.91      0.91      0.89      0.78       156
              1       0.74      0.91      0.87      0.81      0.89      0.79        65
    
    avg / total       0.89      0.88      0.90      0.88      0.89      0.78       221
    
    

## Naive Bayes Classifier


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```




    GaussianNB()




```python
# Calculated the balanced accuracy score
y_pred = gnb.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.5698717948717948



## XGBoost


```python
from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric='mlogloss', gamma=0, gpu_id=-1,
                  grow_policy='depthwise', importance_type=None,
                  interaction_constraints='', learning_rate=0.300000012,
                  max_bin=256, max_cat_to_onehot=4, max_delta_step=0, max_depth=6,
                  max_leaves=0, min_child_weight=1, missing=nan,
                  monotone_constraints='()', n_estimators=100, n_jobs=0,
                  num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, ...)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.9814102564102565




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Priority", "Actual Low-Priority"],
    columns=["Predicted High-Priority", "Predicted Low-Priority"]
)

# Displaying results
display(cm_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Priority</th>
      <th>Predicted Low-Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Priority</th>
      <td>155</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Actual Low-Priority</th>
      <td>2</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


     





