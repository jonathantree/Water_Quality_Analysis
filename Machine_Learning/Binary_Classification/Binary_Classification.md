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
df.sample(20)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county_FIPS</th>
      <th>Geographic_Area_Name</th>
      <th>County</th>
      <th>GEOID</th>
      <th>Total_Population</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific_Islander</th>
      <th>...</th>
      <th>Shannon_Race_DI</th>
      <th>Shannon_Ethnic_DI</th>
      <th>Gini_Index</th>
      <th>County_FIPS</th>
      <th>Num_Contaminants</th>
      <th>Sum_Population_Served</th>
      <th>Sum_ContaminantFactor</th>
      <th>Min_Contaminant_Factor</th>
      <th>Max_Contaminant_Factor</th>
      <th>Avg_Contaminant_Factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>559</th>
      <td>26133</td>
      <td>Osceola County, Michigan</td>
      <td>Osceola County</td>
      <td>0500000US26133</td>
      <td>22891</td>
      <td>21414</td>
      <td>166</td>
      <td>151</td>
      <td>35</td>
      <td>7</td>
      <td>...</td>
      <td>0.378641</td>
      <td>0.088039</td>
      <td>0.4086</td>
      <td>26133</td>
      <td>3</td>
      <td>5719</td>
      <td>1507</td>
      <td>266</td>
      <td>856</td>
      <td>502.33</td>
    </tr>
    <tr>
      <th>500</th>
      <td>26007</td>
      <td>Alpena County, Michigan</td>
      <td>Alpena County</td>
      <td>0500000US26007</td>
      <td>28907</td>
      <td>27177</td>
      <td>100</td>
      <td>123</td>
      <td>120</td>
      <td>13</td>
      <td>...</td>
      <td>0.348201</td>
      <td>0.075467</td>
      <td>0.4332</td>
      <td>26007</td>
      <td>3</td>
      <td>15930</td>
      <td>3315</td>
      <td>1109</td>
      <td>804</td>
      <td>1105.00</td>
    </tr>
    <tr>
      <th>663</th>
      <td>29221</td>
      <td>Washington County, Missouri</td>
      <td>Washington County</td>
      <td>0500000US29221</td>
      <td>23514</td>
      <td>21465</td>
      <td>611</td>
      <td>80</td>
      <td>41</td>
      <td>2</td>
      <td>...</td>
      <td>0.431321</td>
      <td>0.054405</td>
      <td>0.4966</td>
      <td>29221</td>
      <td>8</td>
      <td>6463</td>
      <td>1762</td>
      <td>0</td>
      <td>5</td>
      <td>220.25</td>
    </tr>
    <tr>
      <th>527</th>
      <td>26065</td>
      <td>Ingham County, Michigan</td>
      <td>Ingham County</td>
      <td>0500000US26065</td>
      <td>284900</td>
      <td>198552</td>
      <td>35581</td>
      <td>1536</td>
      <td>16523</td>
      <td>125</td>
      <td>...</td>
      <td>1.231899</td>
      <td>0.292525</td>
      <td>0.4737</td>
      <td>26065</td>
      <td>5</td>
      <td>44333</td>
      <td>1503</td>
      <td>148</td>
      <td>808</td>
      <td>300.60</td>
    </tr>
    <tr>
      <th>275</th>
      <td>55087</td>
      <td>Outagamie County, Wisconsin</td>
      <td>Outagamie County</td>
      <td>0500000US55087</td>
      <td>190705</td>
      <td>164009</td>
      <td>3054</td>
      <td>3144</td>
      <td>6619</td>
      <td>113</td>
      <td>...</td>
      <td>0.765223</td>
      <td>0.196778</td>
      <td>0.4149</td>
      <td>55087</td>
      <td>7</td>
      <td>134452</td>
      <td>5730</td>
      <td>372</td>
      <td>1541</td>
      <td>818.57</td>
    </tr>
    <tr>
      <th>422</th>
      <td>17155</td>
      <td>Putnam County, Illinois</td>
      <td>Putnam County</td>
      <td>0500000US17155</td>
      <td>5637</td>
      <td>5175</td>
      <td>27</td>
      <td>14</td>
      <td>19</td>
      <td>5</td>
      <td>...</td>
      <td>0.550279</td>
      <td>0.231715</td>
      <td>0.3815</td>
      <td>17155</td>
      <td>5</td>
      <td>2910</td>
      <td>3754</td>
      <td>110</td>
      <td>444</td>
      <td>750.80</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38101</td>
      <td>Ward County, North Dakota</td>
      <td>Ward County</td>
      <td>0500000US38101</td>
      <td>69919</td>
      <td>57038</td>
      <td>3025</td>
      <td>1707</td>
      <td>1121</td>
      <td>134</td>
      <td>...</td>
      <td>0.927601</td>
      <td>0.242232</td>
      <td>0.4314</td>
      <td>38101</td>
      <td>7</td>
      <td>3248</td>
      <td>7497</td>
      <td>557</td>
      <td>1978</td>
      <td>1071.00</td>
    </tr>
    <tr>
      <th>730</th>
      <td>41039</td>
      <td>Lane County, Oregon</td>
      <td>Lane County</td>
      <td>0500000US41039</td>
      <td>382971</td>
      <td>309194</td>
      <td>4661</td>
      <td>4675</td>
      <td>9621</td>
      <td>1016</td>
      <td>...</td>
      <td>0.975918</td>
      <td>0.322106</td>
      <td>0.4620</td>
      <td>41039</td>
      <td>3</td>
      <td>12901</td>
      <td>468</td>
      <td>202</td>
      <td>46</td>
      <td>156.00</td>
    </tr>
    <tr>
      <th>584</th>
      <td>29015</td>
      <td>Benton County, Missouri</td>
      <td>Benton County</td>
      <td>0500000US29015</td>
      <td>19394</td>
      <td>18031</td>
      <td>77</td>
      <td>109</td>
      <td>58</td>
      <td>3</td>
      <td>...</td>
      <td>0.386758</td>
      <td>0.081969</td>
      <td>0.4961</td>
      <td>29015</td>
      <td>21</td>
      <td>4070</td>
      <td>419</td>
      <td>0</td>
      <td>97</td>
      <td>19.95</td>
    </tr>
    <tr>
      <th>462</th>
      <td>20073</td>
      <td>Greenwood County, Kansas</td>
      <td>Greenwood County</td>
      <td>0500000US20073</td>
      <td>6016</td>
      <td>5494</td>
      <td>29</td>
      <td>36</td>
      <td>37</td>
      <td>2</td>
      <td>...</td>
      <td>0.501460</td>
      <td>0.148080</td>
      <td>0.4675</td>
      <td>20073</td>
      <td>7</td>
      <td>3955</td>
      <td>10918</td>
      <td>1386</td>
      <td>918</td>
      <td>1559.71</td>
    </tr>
    <tr>
      <th>232</th>
      <td>50007</td>
      <td>Chittenden County, Vermont</td>
      <td>Chittenden County</td>
      <td>0500000US50007</td>
      <td>168323</td>
      <td>144237</td>
      <td>4911</td>
      <td>385</td>
      <td>7244</td>
      <td>54</td>
      <td>...</td>
      <td>0.698324</td>
      <td>0.128519</td>
      <td>0.4476</td>
      <td>50007</td>
      <td>4</td>
      <td>136627</td>
      <td>4345</td>
      <td>714</td>
      <td>1795</td>
      <td>1086.25</td>
    </tr>
    <tr>
      <th>839</th>
      <td>54081</td>
      <td>Raleigh County, West Virginia</td>
      <td>Raleigh County</td>
      <td>0500000US54081</td>
      <td>74591</td>
      <td>64025</td>
      <td>5848</td>
      <td>137</td>
      <td>832</td>
      <td>17</td>
      <td>...</td>
      <td>0.627998</td>
      <td>0.082230</td>
      <td>0.4694</td>
      <td>54081</td>
      <td>6</td>
      <td>60193</td>
      <td>22485</td>
      <td>1306</td>
      <td>1800</td>
      <td>3747.50</td>
    </tr>
    <tr>
      <th>570</th>
      <td>26155</td>
      <td>Shiawassee County, Michigan</td>
      <td>Shiawassee County</td>
      <td>0500000US26155</td>
      <td>68094</td>
      <td>63132</td>
      <td>326</td>
      <td>264</td>
      <td>298</td>
      <td>23</td>
      <td>...</td>
      <td>0.438506</td>
      <td>0.133780</td>
      <td>0.4192</td>
      <td>26155</td>
      <td>16</td>
      <td>12228</td>
      <td>10273</td>
      <td>1218</td>
      <td>957</td>
      <td>642.06</td>
    </tr>
    <tr>
      <th>860</th>
      <td>6047</td>
      <td>Merced County, California</td>
      <td>Merced County</td>
      <td>0500000US06047</td>
      <td>281202</td>
      <td>104534</td>
      <td>9159</td>
      <td>7519</td>
      <td>20716</td>
      <td>808</td>
      <td>...</td>
      <td>1.752052</td>
      <td>0.664908</td>
      <td>0.4608</td>
      <td>6047</td>
      <td>3</td>
      <td>1604</td>
      <td>4808</td>
      <td>1192</td>
      <td>1886</td>
      <td>1602.67</td>
    </tr>
    <tr>
      <th>792</th>
      <td>42097</td>
      <td>Northumberland County, Pennsylvania</td>
      <td>Northumberland County</td>
      <td>0500000US42097</td>
      <td>91647</td>
      <td>82821</td>
      <td>2580</td>
      <td>170</td>
      <td>458</td>
      <td>25</td>
      <td>...</td>
      <td>0.582091</td>
      <td>0.184416</td>
      <td>0.4314</td>
      <td>42097</td>
      <td>15</td>
      <td>73675</td>
      <td>11182</td>
      <td>0</td>
      <td>90</td>
      <td>745.47</td>
    </tr>
    <tr>
      <th>861</th>
      <td>6049</td>
      <td>Modoc County, California</td>
      <td>Modoc County</td>
      <td>0500000US06049</td>
      <td>8700</td>
      <td>6772</td>
      <td>68</td>
      <td>447</td>
      <td>62</td>
      <td>14</td>
      <td>...</td>
      <td>1.105380</td>
      <td>0.413427</td>
      <td>0.3965</td>
      <td>6049</td>
      <td>4</td>
      <td>4082</td>
      <td>5845</td>
      <td>102</td>
      <td>2400</td>
      <td>1461.25</td>
    </tr>
    <tr>
      <th>726</th>
      <td>41031</td>
      <td>Jefferson County, Oregon</td>
      <td>Jefferson County</td>
      <td>0500000US41031</td>
      <td>24502</td>
      <td>15992</td>
      <td>146</td>
      <td>3324</td>
      <td>150</td>
      <td>25</td>
      <td>...</td>
      <td>1.401538</td>
      <td>0.506097</td>
      <td>0.4619</td>
      <td>41031</td>
      <td>3</td>
      <td>18540</td>
      <td>3028</td>
      <td>1815</td>
      <td>708</td>
      <td>1009.33</td>
    </tr>
    <tr>
      <th>499</th>
      <td>26005</td>
      <td>Allegan County, Michigan</td>
      <td>Allegan County</td>
      <td>0500000US26005</td>
      <td>120502</td>
      <td>105564</td>
      <td>1585</td>
      <td>800</td>
      <td>875</td>
      <td>23</td>
      <td>...</td>
      <td>0.731340</td>
      <td>0.273649</td>
      <td>0.4069</td>
      <td>26005</td>
      <td>9</td>
      <td>103158</td>
      <td>10796</td>
      <td>112</td>
      <td>296</td>
      <td>1199.56</td>
    </tr>
    <tr>
      <th>403</th>
      <td>17111</td>
      <td>McHenry County, Illinois</td>
      <td>McHenry County</td>
      <td>0500000US17111</td>
      <td>310229</td>
      <td>247894</td>
      <td>4285</td>
      <td>1535</td>
      <td>8804</td>
      <td>62</td>
      <td>...</td>
      <td>1.044129</td>
      <td>0.423662</td>
      <td>0.4076</td>
      <td>17111</td>
      <td>21</td>
      <td>287367</td>
      <td>14596</td>
      <td>1591</td>
      <td>997</td>
      <td>695.05</td>
    </tr>
    <tr>
      <th>757</th>
      <td>42027</td>
      <td>Centre County, Pennsylvania</td>
      <td>Centre County</td>
      <td>0500000US42027</td>
      <td>158172</td>
      <td>131212</td>
      <td>5477</td>
      <td>203</td>
      <td>11405</td>
      <td>46</td>
      <td>...</td>
      <td>0.795779</td>
      <td>0.154118</td>
      <td>0.4714</td>
      <td>42027</td>
      <td>27</td>
      <td>143387</td>
      <td>4772</td>
      <td>0</td>
      <td>92</td>
      <td>176.74</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 37 columns</p>
</div>




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
