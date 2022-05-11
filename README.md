# 2022 Data Analytics Final Project
## Identifying at-risk and underserved communities with poor quality drinking water in the United States

## Project Overview
The purpose of this project is to analyze drinking water quality to determine if the quality of community's drinking water is correlated with certain demographic markers, such as income level. Specifically, we are aiming to use socioeconomic data and drinking water quality data to identify at-risk communities that are historically underserved. By identifying which communities are at most risk (i.e., poverty level, racial inequity, average house age >60 years, etc.) and filtering those communities based on our analysis of water quality data, we can target communities most in need of humanitarian support to remediate their water source.

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
Data is being sourced from web scraping of [Environmental Working Group's Tap Water Database](https://www.ewg.org/tapwater/), usgs.gov, epa.gov, and the US Census data (2010).

## Machine Learning Mock-up

### Machine Learning Overview
The machine learning scope of this project will be three-fold: one supervised machine learning model trained on water quality data that will predict the quality of any input data from a water utility report, another unsupervised categorization model which will inform categories of interest to develop a supervised model. The latter models will train on the datasets which have been scraped from the [Environmental Working Group's Tap Water Database](https://www.ewg.org/tapwater/) as well as socioeconomic data retrieved from the US Census API. The purpose of having this three-fold structure is to be able to identify at-risk communities due to socioeconomic inequity with high level pollutants in their drinking water. The first model will then be able to take as input a water quality report and predict a quality index for any communities identified from the other model.

### Water Quality Binary Classifier Prediction Model

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.read_csv('Kaggle_EDA/water_potability.csv')
```


```python
data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ph</th>
      <th>Hardness</th>
      <th>Solids</th>
      <th>Chloramines</th>
      <th>Sulfate</th>
      <th>Conductivity</th>
      <th>Organic_carbon</th>
      <th>Trihalomethanes</th>
      <th>Turbidity</th>
      <th>Potability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>204.890455</td>
      <td>20791.318981</td>
      <td>7.300212</td>
      <td>368.516441</td>
      <td>564.308654</td>
      <td>10.379783</td>
      <td>86.990970</td>
      <td>2.963135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.716080</td>
      <td>129.422921</td>
      <td>18630.057858</td>
      <td>6.635246</td>
      <td>NaN</td>
      <td>592.885359</td>
      <td>15.180013</td>
      <td>56.329076</td>
      <td>4.500656</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.099124</td>
      <td>224.236259</td>
      <td>19909.541732</td>
      <td>9.275884</td>
      <td>NaN</td>
      <td>418.606213</td>
      <td>16.868637</td>
      <td>66.420093</td>
      <td>3.055934</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.316766</td>
      <td>214.373394</td>
      <td>22018.417441</td>
      <td>8.059332</td>
      <td>356.886136</td>
      <td>363.266516</td>
      <td>18.436524</td>
      <td>100.341674</td>
      <td>4.628771</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.092223</td>
      <td>181.101509</td>
      <td>17978.986339</td>
      <td>6.546600</td>
      <td>310.135738</td>
      <td>398.410813</td>
      <td>11.558279</td>
      <td>31.997993</td>
      <td>4.075075</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3271</th>
      <td>4.668102</td>
      <td>193.681735</td>
      <td>47580.991603</td>
      <td>7.166639</td>
      <td>359.948574</td>
      <td>526.424171</td>
      <td>13.894419</td>
      <td>66.687695</td>
      <td>4.435821</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3272</th>
      <td>7.808856</td>
      <td>193.553212</td>
      <td>17329.802160</td>
      <td>8.061362</td>
      <td>NaN</td>
      <td>392.449580</td>
      <td>19.903225</td>
      <td>NaN</td>
      <td>2.798243</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3273</th>
      <td>9.419510</td>
      <td>175.762646</td>
      <td>33155.578218</td>
      <td>7.350233</td>
      <td>NaN</td>
      <td>432.044783</td>
      <td>11.039070</td>
      <td>69.845400</td>
      <td>3.298875</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3274</th>
      <td>5.126763</td>
      <td>230.603758</td>
      <td>11983.869376</td>
      <td>6.303357</td>
      <td>NaN</td>
      <td>402.883113</td>
      <td>11.168946</td>
      <td>77.488213</td>
      <td>4.708658</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3275</th>
      <td>7.874671</td>
      <td>195.102299</td>
      <td>17404.177061</td>
      <td>7.509306</td>
      <td>NaN</td>
      <td>327.459760</td>
      <td>16.140368</td>
      <td>78.698446</td>
      <td>2.309149</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3276 rows × 10 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3276 entries, 0 to 3275
    Data columns (total 10 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   ph               2785 non-null   float64
     1   Hardness         3276 non-null   float64
     2   Solids           3276 non-null   float64
     3   Chloramines      3276 non-null   float64
     4   Sulfate          2495 non-null   float64
     5   Conductivity     3276 non-null   float64
     6   Organic_carbon   3276 non-null   float64
     7   Trihalomethanes  3114 non-null   float64
     8   Turbidity        3276 non-null   float64
     9   Potability       3276 non-null   int64  
    dtypes: float64(9), int64(1)
    memory usage: 256.1 KB
    


```python
data.isnull().sum()
```




    ph                 491
    Hardness             0
    Solids               0
    Chloramines          0
    Sulfate            781
    Conductivity         0
    Organic_carbon       0
    Trihalomethanes    162
    Turbidity            0
    Potability           0
    dtype: int64




```python
data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ph</th>
      <th>Hardness</th>
      <th>Solids</th>
      <th>Chloramines</th>
      <th>Sulfate</th>
      <th>Conductivity</th>
      <th>Organic_carbon</th>
      <th>Trihalomethanes</th>
      <th>Turbidity</th>
      <th>Potability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
      <td>3276.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.080795</td>
      <td>196.369496</td>
      <td>22014.092526</td>
      <td>7.122277</td>
      <td>333.775777</td>
      <td>426.205111</td>
      <td>14.284970</td>
      <td>66.396293</td>
      <td>3.966786</td>
      <td>0.390110</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.469956</td>
      <td>32.879761</td>
      <td>8768.570828</td>
      <td>1.583085</td>
      <td>36.142612</td>
      <td>80.824064</td>
      <td>3.308162</td>
      <td>15.769881</td>
      <td>0.780382</td>
      <td>0.487849</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>47.432000</td>
      <td>320.942611</td>
      <td>0.352000</td>
      <td>129.000000</td>
      <td>181.483754</td>
      <td>2.200000</td>
      <td>0.738000</td>
      <td>1.450000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.277673</td>
      <td>176.850538</td>
      <td>15666.690297</td>
      <td>6.127421</td>
      <td>317.094638</td>
      <td>365.734414</td>
      <td>12.065801</td>
      <td>56.647656</td>
      <td>3.439711</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.080795</td>
      <td>196.967627</td>
      <td>20927.833607</td>
      <td>7.130299</td>
      <td>333.775777</td>
      <td>421.884968</td>
      <td>14.218338</td>
      <td>66.396293</td>
      <td>3.955028</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.870050</td>
      <td>216.667456</td>
      <td>27332.762127</td>
      <td>8.114887</td>
      <td>350.385756</td>
      <td>481.792304</td>
      <td>16.557652</td>
      <td>76.666609</td>
      <td>4.500320</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.000000</td>
      <td>323.124000</td>
      <td>61227.196008</td>
      <td>13.127000</td>
      <td>481.030642</td>
      <td>753.342620</td>
      <td>28.300000</td>
      <td>124.000000</td>
      <td>6.739000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning

### Replace all null pH, Sulfate, and Trihalomethanes values with the mean


```python
data.fillna(data.mean(), inplace=True)
```


```python
data.isnull().sum()
```




    ph                 0
    Hardness           0
    Solids             0
    Chloramines        0
    Sulfate            0
    Conductivity       0
    Organic_carbon     0
    Trihalomethanes    0
    Turbidity          0
    Potability         0
    dtype: int64



## EDA


```python
#Check to see if we can do dimensionality reduction
#Use the heatmap plot to see if we have any strong correlations and if so we can drop that feature
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='Spectral' )
```




    <AxesSubplot:>




    
![png](kaggle_waterquality_ML_dev_basic_files/kaggle_waterquality_ML_dev_basic_11_1.png)
    


### As seen in the plot above, there are no feature correlations that have a correlation coefficient higher than 0.082 which comes from the pH and solids

## Take a look at the data distributions


```python
sns.set_theme(style="ticks")

# Initialize the figure 
f, ax = plt.subplots(figsize=(14, 6))

# Plot the orbital period with horizontal boxes
sns.boxplot(data=data, whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(data=data, size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
```


    
![png](kaggle_waterquality_ML_dev_basic_files/kaggle_waterquality_ML_dev_basic_14_0.png)
    


### The solids show a large distribution outside of the 1st and 3rd quantiles, we will keep these outliers to improve the accuracy of our model


```python
data.columns
```




    Index(['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
           'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'],
          dtype='object')




```python
# Shows KDEs instead of histograms along the diagonal and check for normality in distributions of the features
sns.pairplot(data, hue="Potability")
```




    <seaborn.axisgrid.PairGrid at 0x181994105e0>




    
![png](kaggle_waterquality_ML_dev_basic_files/kaggle_waterquality_ML_dev_basic_17_1.png)
    


## Check for data imbalance


```python
data.Potability.value_counts()
```




    0    1998
    1    1278
    Name: Potability, dtype: int64




```python
sns.countplot(data.Potability)
```

    C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='Potability', ylabel='count'>




    
![png](kaggle_waterquality_ML_dev_basic_files/kaggle_waterquality_ML_dev_basic_20_2.png)
    


### In this prepatory example dataset, we actually have more bad quality than good, but the data is pretty well balanced

### Check again in simple panbas hist plot for normality and see if we need to complete a normalization step before we partition the data


```python
data.hist(figsize=(14,12))
plt.show()
```


    
![png](kaggle_waterquality_ML_dev_basic_files/kaggle_waterquality_ML_dev_basic_23_0.png)
    


## Data Partitioning


```python
#Input features
X = data.drop('Potability', axis=1)
#Target
y = data.Potability 
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True )
```


```python
y_train
```




    675     1
    1359    0
    1391    0
    1727    0
    1677    0
           ..
    2763    1
    905     0
    1096    1
    235     0
    1061    0
    Name: Potability, Length: 2620, dtype: int64



## Model Training

### Decision tree binary classifier


```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
```


```python
dt.fit(X_train, y_train)
```




    DecisionTreeClassifier()




```python
y_prediction = dt.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score, confusion_matrix
```


```python
accuracy_score(y_prediction, y_test)*100
```




    60.0609756097561




```python
confusion_matrix(y_prediction, y_test)
```




    array([[254, 143],
           [119, 140]], dtype=int64)



## Initial Training of this model resulted in a False-Negative value of 119 which is undesirable

### Other possible models to test will be:
1. Naive Bayes
2. Logistic Regression
3. K-Nearest Neighbours
4. Support Vector Machine
5. Decision Tree
6. Bagging Decision Tree (Ensemble Learning I)
7. Boosted Decision Tree (Ensemble Learning II)
8. Random Forest (Ensemble Learning III)
9. Voting Classification (Ensemble Learning IV)
10. Neural Network (Deep Learning)

## Model optimization and hyperparameter tuning


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
```


```python
dt = DecisionTreeClassifier()
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
min_samples_split=range(1,11)
parameters = dict(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
grid_search_cv_dt = GridSearchCV(estimator=dt, param_grid=parameters, scoring='accuracy', cv=cv)
```


```python
grid_search_cv_dt.fit(X_train,y_train)
print(grid_search_cv_dt.best_params_)
```

    {'criterion': 'entropy', 'min_samples_split': 9, 'splitter': 'best'}
    

    C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\sklearn\model_selection\_validation.py:372: FitFailedWarning: 
    200 fits failed out of a total of 2000.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    200 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\sklearn\model_selection\_validation.py", line 680, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\sklearn\tree\_classes.py", line 937, in fit
        super().fit(
      File "C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\sklearn\tree\_classes.py", line 250, in fit
        raise ValueError(
    ValueError: min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0]; got the integer 1
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    C:\Users\jonat\anaconda3\envs\mlenv\lib\site-packages\sklearn\model_selection\_search.py:969: UserWarning: One or more of the test scores are non-finite: [       nan        nan 0.58312977 0.57977099 0.58652672 0.5798855
     0.5859542  0.58125954 0.58572519 0.58251908 0.58675573 0.58843511
     0.58801527 0.58782443 0.58816794 0.58164122 0.58885496 0.5880916
     0.58942748 0.58793893        nan        nan 0.59076336 0.57820611
     0.59007634 0.57828244 0.58919847 0.57912214 0.59152672 0.5880916
     0.59320611 0.58580153 0.59248092 0.58549618 0.59301527 0.58675573
     0.5948855  0.58629771 0.59477099 0.58725191]
      warnings.warn(
    


```python
prediction_grid = grid_search_cv_dt.predict(X_test)
```


```python
accuracy_score(y_test, prediction_grid)*100
```




    59.14634146341463




```python
confusion_matrix(y_test,prediction_grid)
```




    array([[269, 104],
           [164, 119]], dtype=int64)



## Oddly enough, this did not perform well

### Future testing will include other models and optimizations

#### Other Models for the EWG and socioeconomic data will be constructed once that dataset is complete. The EWG scraping has been successful and the remaining socioeconomic data needs to be joined from the US Census API retrieval.

## Other Datasets scraped from [Environmental Working Group's Tap Water Database](https://www.ewg.org/tapwater/)
The following illustrates an example web scrape of all of the utilities in one zip code. This will be executed for a list of all the zip codes in a state and a dataset for each state will be added to our database.

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as soup
import pandas as pd
import requests

```


```python
driver = webdriver.Chrome()
driver.get("https://www.ewg.org/tapwater/")
```


```python
element = driver.find_element(By.XPATH,'/html/body/main/section[1]/form/input[1]')

```


```python
element.send_keys('97045')
```


```python
button = driver.find_element(By.XPATH,'/html/body/main/section[1]/form/input[3]').click()

```


```python
zip_url = driver.current_url
zip_url
```




    'https://www.ewg.org/tapwater/search-results.php?zip5=97045&searchtype=zip'




```python
url = requests.get(zip_url)
table = pd.read_html(url.text)
```





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Utility name</th>
      <th>City</th>
      <th>People served</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>North Clackamas County Water Commission</td>
      <td>Oregon City, OR</td>
      <td>Population served:87,700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South Fork Water Board</td>
      <td>Oregon City, OR</td>
      <td>Population served:65,000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sunrise Water Authority</td>
      <td>Happy Valley, OR</td>
      <td>Population served:46,228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clackamas River Water - Clackamas</td>
      <td>Clackamas, OR</td>
      <td>Population served:41,338</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lake Oswego Municipal Water</td>
      <td>Lake Oswego, OR</td>
      <td>Population served:36,453</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Oregon City</td>
      <td>Oregon City, OR</td>
      <td>Population served:33,940</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Oak Lodge Water District</td>
      <td>Milwaukie, OR</td>
      <td>Population served:30,000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>City of West Linn</td>
      <td>West Linn, OR</td>
      <td>Population served:25,002</td>
    </tr>
    <tr>
      <th>8</th>
      <td>City of Milwaukie</td>
      <td>Milwaukie, OR</td>
      <td>Population served:20,500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>City of Wilsonville</td>
      <td>Wilsonville, OR</td>
      <td>Population served:22,729</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_utilities = len(table[0])
num_utilities
```




    10




```python
#Initialize an empty list to contain all of our data
contaminant_list = []
contaminants_column = []
for utility_vist in range(num_utilities):
    print(utility_vist)
    utility_vist = str(utility_vist+1)
    XPath = "/html/body/div[3]/main/figure[1]/table/tbody/tr["+utility_vist+"]/td[1]/a"
    selection = driver.find_element(By.XPATH, XPath).click()
    # Parse the resulting html with soup
    page_source = driver.page_source
    html_soup = soup(page_source, 'html.parser')
    
    #Get the name of the Water Utility
    Utility = html_soup.find('h1').text
    Utility
    
    #get the html data we need
    data_box = html_soup.find_all('div', class_='contaminant-name')
    
    for i in range(len(data_box)):
        data = data_box[i].find_all('span')
        data_measure = []
        d = {
            'Utility' : Utility,
            'Contaminant': '', 
            'Utility Measuremnt':'', 
            'EWG HEALTH GUIDELINE': '',
            'Legal Limit':'' 
        }
        
        contaminant_name = data_box[i].find('h3')
        d['Contaminant'] = contaminant_name
        
        for j in range(len(data)):
            measurement = data[j].text
            #print(measurement)
            data_measure.append(measurement)
            #print(data_measure)

        try:
            d['Utility Measuremnt'] = data_measure[data_measure.index('THIS UTILITY')+1]
        except ValueError:
            print("A value error arose")
        except:
            print("Something else went wrong")
        try:
            d['EWG HEALTH GUIDELINE'] = data_measure[data_measure.index('EWG HEALTH GUIDELINE')+1]
        except ValueError:
            print("A value error arose")
        except:
            print("Something else went wrong") 
        try:
            d['Legal Limit'] = data_measure[data_measure.index('LEGAL LIMIT')+1]
        except ValueError:
            print("A value error arose")
        except:
            print("Something else went wrong") 

        contaminant_list.append(d)
        

  
    driver.back()

```



```python
df = pd.DataFrame(contaminant_list)
```


```python
pd.set_option("display.max_rows", None, "display.max_columns", None)

df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Utility</th>
      <th>Contaminant</th>
      <th>Utility Measuremnt</th>
      <th>EWG HEALTH GUIDELINE</th>
      <th>Legal Limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Haloacetic acids (HAA5)†]</td>
      <td>14.0 ppb</td>
      <td>0.1 ppb</td>
      <td>60 ppb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Total trihalomethanes (TTHMs)†]</td>
      <td>16.0 ppb</td>
      <td>0.15 ppb</td>
      <td>80 ppb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Barium]</td>
      <td>1.77 ppb</td>
      <td>700 ppb</td>
      <td>2,000 ppb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Nitrate]</td>
      <td>0.190 ppm</td>
      <td>0.14 ppm</td>
      <td>10 ppm</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Nitrate and nitrite]</td>
      <td>0.0720 ppm</td>
      <td>0.14 ppm</td>
      <td>10 ppm</td>
    </tr>
    <tr>
      <th>5</th>
      <td>North Clackamas County Water Commission</td>
      <td>[Nitrite]</td>
      <td>0.00778 ppm</td>
      <td></td>
      <td>1 ppm</td>
    </tr>
    <tr>
      <th>6</th>
      <td>North Clackamas County Water Commission</td>
      <td>None</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>South Fork Water Board</td>
      <td>[Haloacetic acids (HAA5)†]</td>
      <td>21.6 ppb</td>
      <td>0.1 ppb</td>
      <td>60 ppb</td>
    </tr>
    <tr>
      <th>8</th>
      <td>South Fork Water Board</td>
      <td>[Nitrate]</td>
      <td>0.286 ppm</td>
      <td>0.14 ppm</td>
      <td>10 ppm</td>
    </tr>
    <tr>
      <th>9</th>
      <td>South Fork Water Board</td>
      <td>[Total trihalomethanes (TTHMs)†]</td>
      <td>29.8 ppb</td>
      <td>0.15 ppb</td>
      <td>80 ppb</td>
    </tr>
    <tr>
      <th>10</th>
      <td>South Fork Water Board</td>
      <td>[Barium]</td>
      <td>2.89 ppb</td>
      <td>700 ppb</td>
      <td>2,000 ppb</td>
    </tr>
    <tr>
      <th>11</th>
      <td>South Fork Water Board</td>
      <td>[Fluoride]</td>
      <td>0.0286 ppm</td>
      <td></td>
      <td>4 ppm</td>
    </tr>
    <tr>
      <th>12</th>
      <td>South Fork Water Board</td>
      <td>[Nitrate and nitrite]</td>
      <td>0.250 ppm</td>
      <td>0.14 ppm</td>
      <td>10 ppm</td>
    </tr>
       <th>125</th>
      <td>City of Wilsonville</td>
      <td>[Barium]</td>
      <td>5.50 ppb</td>
      <td>700 ppb</td>
      <td>2,000 ppb</td>
    </tr>
    <tr>
      <th>126</th>
      <td>City of Wilsonville</td>
      <td>[Chlorate]</td>
      <td>92.3 ppb</td>
      <td>210 ppb</td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>City of Wilsonville</td>
      <td>[Chromium (total)]</td>
      <td>0.00588 ppb</td>
      <td></td>
      <td>100 ppb</td>
    </tr>
    <tr>
      <th>128</th>
      <td>City of Wilsonville</td>
      <td>[Manganese]</td>
      <td>3.12 ppb</td>
      <td>100 ppb</td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td>City of Wilsonville</td>
      <td>[Strontium]</td>
      <td>0.0380 ppb</td>
      <td>1,500 ppb</td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td>City of Wilsonville</td>
      <td>[Vanadium]</td>
      <td>1.59 ppb</td>
      <td>21 ppb</td>
      <td></td>
    </tr>
    <tr>
      <th>131</th>
      <td>City of Wilsonville</td>
      <td>None</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
driver.quit()
```


```python
df.to_csv('Resources/Data/97045_scrape.csv')
```


```python

```

## Database Design Mock-up
We are choosing to use sqalchemy to communicate with our Postgres database. Below is the example construction of one of the tables which we will pull zip codes from to automate more web scraping data.

```python
from config import username, password
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from getpass import getpass
import pandas as pd

url = f'postgresql://{username}:{password}@localhost:5432/project_db'
```


```python
# Create an engine object.
engine = create_engine(url, echo=True)

# Create database if it does not exist.
if not database_exists(engine.url):
    create_database(engine.url)
    connection = engine.connect()
else:
    # Connect the database if exists.
    connection = engine.connect()
```

    2022-05-07 11:25:36,760 INFO sqlalchemy.engine.Engine select pg_catalog.version()
    2022-05-07 11:25:36,761 INFO sqlalchemy.engine.Engine [raw sql] {}
    2022-05-07 11:25:36,763 INFO sqlalchemy.engine.Engine select current_schema()
    2022-05-07 11:25:36,763 INFO sqlalchemy.engine.Engine [raw sql] {}
    2022-05-07 11:25:36,765 INFO sqlalchemy.engine.Engine show standard_conforming_strings
    2022-05-07 11:25:36,766 INFO sqlalchemy.engine.Engine [raw sql] {}
    


```python
df = pd.read_csv("ZIP_Locale_Detail_1_tbl.csv", low_memory=False)
df.to_sql('usps', con=engine, if_exists='append')    
```

    2022-05-07 11:32:13,375 INFO sqlalchemy.engine.Engine select relname from pg_class c join pg_namespace n on n.oid=c.relnamespace where pg_catalog.pg_table_is_visible(c.oid) and relname=%(name)s
    2022-05-07 11:32:13,375 INFO sqlalchemy.engine.Engine [generated in 0.00093s] {'name': 'usps'}
    2022-05-07 11:32:13,385 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:32:13,387 INFO sqlalchemy.engine.Engine 
    CREATE TABLE usps (
    	index BIGINT, 
    	"AREA NAME" TEXT, 
    	"AREA CODE" TEXT, 
    	"DISTRICT NAME" TEXT, 
    	"DISTRICT NO" BIGINT, 
    	"DELIVERY ZIPCODE" BIGINT, 
    	"LOCALE NAME" TEXT, 
    	"PHYSICAL DELV ADDR" TEXT, 
    	"PHYSICAL CITY" TEXT, 
    	"PHYSICAL STATE" TEXT, 
    	"PHYSICAL ZIP" FLOAT(53), 
    	"PHYSICAL ZIP 4" FLOAT(53)
    )
    
    
    2022-05-07 11:32:13,389 INFO sqlalchemy.engine.Engine [no key 0.00161s] {}
    2022-05-07 11:32:13,422 INFO sqlalchemy.engine.Engine CREATE INDEX ix_usps_index ON usps (index)
    2022-05-07 11:32:13,422 INFO sqlalchemy.engine.Engine [no key 0.00086s] {}
    2022-05-07 11:32:13,430 INFO sqlalchemy.engine.Engine COMMIT
    2022-05-07 11:32:13,473 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:32:14,838 INFO sqlalchemy.engine.Engine INSERT INTO usps (index, "AREA NAME", "AREA CODE", "DISTRICT NAME", "DISTRICT NO", "DELIVERY ZIPCODE", "LOCALE NAME", "PHYSICAL DELV ADDR", "PHYSICAL CITY", "PHYSICAL STATE", "PHYSICAL ZIP", "PHYSICAL ZIP 4") VALUES (%(index)s, %(AREA NAME)s, %(AREA CODE)s, %(DISTRICT NAME)s, %(DISTRICT NO)s, %(DELIVERY ZIPCODE)s, %(LOCALE NAME)s, %(PHYSICAL DELV ADDR)s, %(PHYSICAL CITY)s, %(PHYSICAL STATE)s, %(PHYSICAL ZIP)s, %(PHYSICAL ZIP 4)s)
    2022-05-07 11:32:14,839 INFO sqlalchemy.engine.Engine [generated in 1.24273s] ({'index': 0, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 601, 'LOCALE NAME': 'ADJUNTAS', 'PHYSICAL DELV ADDR': '37 CALLE MUNOZ RIVERA', 'PHYSICAL CITY': 'ADJUNTAS', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 601.0, 'PHYSICAL ZIP 4': 9998.0}, {'index': 1, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 602, 'LOCALE NAME': 'AGUADA', 'PHYSICAL DELV ADDR': '5 AVE NATIVO ALERS', 'PHYSICAL CITY': 'AGUADA', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 602.0, 'PHYSICAL ZIP 4': 9998.0}, {'index': 2, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 603, 'LOCALE NAME': 'AGUADILLA', 'PHYSICAL DELV ADDR': '50 CARR 459 STE 1', 'PHYSICAL CITY': 'AGUADILLA', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 603.0, 'PHYSICAL ZIP 4': 9998.0}, {'index': 3, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 604, 'LOCALE NAME': 'RAMEY', 'PHYSICAL DELV ADDR': '100 AVE BORINQUEN', 'PHYSICAL CITY': 'AGUADILLA', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 603.0, 'PHYSICAL ZIP 4': 9996.0}, {'index': 4, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 605, 'LOCALE NAME': 'AGUADILLA', 'PHYSICAL DELV ADDR': '50 CARR 459 STE 1', 'PHYSICAL CITY': 'AGUADILLA', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 603.0, 'PHYSICAL ZIP 4': 9998.0}, {'index': 5, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 605, 'LOCALE NAME': 'VICTORIA', 'PHYSICAL DELV ADDR': '68 CALLE RAMON E BETANCES', 'PHYSICAL CITY': 'AGUADILLA', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 603.0, 'PHYSICAL ZIP 4': 9997.0}, {'index': 6, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 606, 'LOCALE NAME': 'MARICAO', 'PHYSICAL DELV ADDR': '9 CALLE ZUZUARREGUI FRNT', 'PHYSICAL CITY': 'MARICAO', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 606.0, 'PHYSICAL ZIP 4': 9998.0}, {'index': 7, 'AREA NAME': 'SOUTHERN', 'AREA CODE': '4G', 'DISTRICT NAME': 'PUERTO RICO', 'DISTRICT NO': 6, 'DELIVERY ZIPCODE': 610, 'LOCALE NAME': 'ANASCO', 'PHYSICAL DELV ADDR': '93 CALLE 65 INFANTERIA STE 1', 'PHYSICAL CITY': 'ANASCO', 'PHYSICAL STATE': 'PR', 'PHYSICAL ZIP': 610.0, 'PHYSICAL ZIP 4': 2947.0}  ... displaying 10 of 43900 total bound parameter sets ...  {'index': 43898, 'AREA NAME': 'WESTPAC', 'AREA CODE': '4E', 'DISTRICT NAME': 'ALASKA', 'DISTRICT NO': 995, 'DELIVERY ZIPCODE': 99928, 'LOCALE NAME': 'WARD COVE', 'PHYSICAL DELV ADDR': '7172 N TONGASS HWY', 'PHYSICAL CITY': 'WARD COVE', 'PHYSICAL STATE': 'AK', 'PHYSICAL ZIP': 99928.0, 'PHYSICAL ZIP 4': 9800.0}, {'index': 43899, 'AREA NAME': 'WESTPAC', 'AREA CODE': '4E', 'DISTRICT NAME': 'ALASKA', 'DISTRICT NO': 995, 'DELIVERY ZIPCODE': 99929, 'LOCALE NAME': 'WRANGELL', 'PHYSICAL DELV ADDR': '112 FEDERAL WAY', 'PHYSICAL CITY': 'WRANGELL', 'PHYSICAL STATE': 'AK', 'PHYSICAL ZIP': 99929.0, 'PHYSICAL ZIP 4': 9800.0})
    2022-05-07 11:32:17,125 INFO sqlalchemy.engine.Engine COMMIT
    


```python
df = pd.read_csv("EWG_violationpoints.csv", low_memory=False)
df.to_sql('ewg', con=engine, if_exists='append')  
```

    2022-05-07 11:40:51,313 INFO sqlalchemy.engine.Engine select relname from pg_class c join pg_namespace n on n.oid=c.relnamespace where pg_catalog.pg_table_is_visible(c.oid) and relname=%(name)s
    2022-05-07 11:40:51,314 INFO sqlalchemy.engine.Engine [cached since 517.9s ago] {'name': 'ewg'}
    2022-05-07 11:40:51,320 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:40:51,322 INFO sqlalchemy.engine.Engine 
    CREATE TABLE ewg (
    	index BIGINT, 
    	"Utility" TEXT, 
    	"Location" TEXT, 
    	"PeopleServed" BIGINT, 
    	"ViolationPoints" BIGINT
    )
    
    
    2022-05-07 11:40:51,323 INFO sqlalchemy.engine.Engine [no key 0.00107s] {}
    2022-05-07 11:40:51,331 INFO sqlalchemy.engine.Engine CREATE INDEX ix_ewg_index ON ewg (index)
    2022-05-07 11:40:51,332 INFO sqlalchemy.engine.Engine [no key 0.00081s] {}
    2022-05-07 11:40:51,337 INFO sqlalchemy.engine.Engine COMMIT
    2022-05-07 11:40:51,343 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:40:51,368 INFO sqlalchemy.engine.Engine INSERT INTO ewg (index, "Utility", "Location", "PeopleServed", "ViolationPoints") VALUES (%(index)s, %(Utility)s, %(Location)s, %(PeopleServed)s, %(ViolationPoints)s)
    2022-05-07 11:40:51,369 INFO sqlalchemy.engine.Engine [generated in 0.02100s] ({'index': 0, 'Utility': 'Dekalb-Jackson Water Supply District', 'Location': 'Ider, AL', 'PeopleServed': 15531, 'ViolationPoints': 30}, {'index': 1, 'Utility': 'North Clarke Water Authority', 'Location': 'Thomasville, AL', 'PeopleServed': 975, 'ViolationPoints': 25}, {'index': 2, 'Utility': 'Cwm Water Authority', 'Location': 'Dickinson, AL', 'PeopleServed': 1722, 'ViolationPoints': 10}, {'index': 3, 'Utility': 'Centre Water & Sewer Board', 'Location': 'Centre, AL', 'PeopleServed': 7050, 'ViolationPoints': 5}, {'index': 4, 'Utility': 'Mid-central Water Authority', 'Location': 'Fulton, AL', 'PeopleServed': 2331, 'ViolationPoints': 5}, {'index': 5, 'Utility': 'Thomasville Water Works & Sewer Board', 'Location': 'Thomasville, AL', 'PeopleServed': 6897, 'ViolationPoints': 5}, {'index': 6, 'Utility': 'Union Grove Utility Board', 'Location': 'Union Grove, AL', 'PeopleServed': 2862, 'ViolationPoints': 2}, {'index': 7, 'Utility': 'Prichard Water Works Board', 'Location': 'Prichard, AL', 'PeopleServed': 32400, 'ViolationPoints': 2}  ... displaying 10 of 1429 total bound parameter sets ...  {'index': 1427, 'Utility': 'Jackson, Town Of', 'Location': 'Jackson, WY', 'PeopleServed': 8000, 'ViolationPoints': 4}, {'index': 1428, 'Utility': 'Cokeville, Town Of', 'Location': 'Cokeville, WY', 'PeopleServed': 535, 'ViolationPoints': 4})
    2022-05-07 11:40:51,421 INFO sqlalchemy.engine.Engine COMMIT
    


```python
df = pd.read_csv("water_potability.csv", low_memory=False)
df.to_sql('train', con=engine, if_exists='append')  
```

    2022-05-07 11:43:29,651 INFO sqlalchemy.engine.Engine select relname from pg_class c join pg_namespace n on n.oid=c.relnamespace where pg_catalog.pg_table_is_visible(c.oid) and relname=%(name)s
    2022-05-07 11:43:29,652 INFO sqlalchemy.engine.Engine [cached since 676.3s ago] {'name': 'train'}
    2022-05-07 11:43:29,657 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:43:29,660 INFO sqlalchemy.engine.Engine 
    CREATE TABLE train (
    	index BIGINT, 
    	ph FLOAT(53), 
    	"Hardness" FLOAT(53), 
    	"Solids" FLOAT(53), 
    	"Chloramines" FLOAT(53), 
    	"Sulfate" FLOAT(53), 
    	"Conductivity" FLOAT(53), 
    	"Organic_carbon" FLOAT(53), 
    	"Trihalomethanes" FLOAT(53), 
    	"Turbidity" FLOAT(53), 
    	"Potability" BIGINT
    )
    
    
    2022-05-07 11:43:29,661 INFO sqlalchemy.engine.Engine [no key 0.00105s] {}
    2022-05-07 11:43:29,664 INFO sqlalchemy.engine.Engine CREATE INDEX ix_train_index ON train (index)
    2022-05-07 11:43:29,664 INFO sqlalchemy.engine.Engine [no key 0.00074s] {}
    2022-05-07 11:43:29,670 INFO sqlalchemy.engine.Engine COMMIT
    2022-05-07 11:43:29,676 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2022-05-07 11:43:29,771 INFO sqlalchemy.engine.Engine INSERT INTO train (index, ph, "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability") VALUES (%(index)s, %(ph)s, %(Hardness)s, %(Solids)s, %(Chloramines)s, %(Sulfate)s, %(Conductivity)s, %(Organic_carbon)s, %(Trihalomethanes)s, %(Turbidity)s, %(Potability)s)
    2022-05-07 11:43:29,772 INFO sqlalchemy.engine.Engine [generated in 0.08316s] ({'index': 0, 'ph': None, 'Hardness': 204.8904554713363, 'Solids': 20791.318980747023, 'Chloramines': 7.300211873184757, 'Sulfate': 368.51644134980336, 'Conductivity': 564.3086541722439, 'Organic_carbon': 10.3797830780847, 'Trihalomethanes': 86.9909704615088, 'Turbidity': 2.9631353806316407, 'Potability': 0}, {'index': 1, 'ph': 3.71608007538699, 'Hardness': 129.42292051494425, 'Solids': 18630.057857970347, 'Chloramines': 6.635245883862, 'Sulfate': None, 'Conductivity': 592.8853591348523, 'Organic_carbon': 15.18001311635726, 'Trihalomethanes': 56.32907628451764, 'Turbidity': 4.500656274942408, 'Potability': 0}, {'index': 2, 'ph': 8.099124189298397, 'Hardness': 224.23625939355776, 'Solids': 19909.541732292397, 'Chloramines': 9.275883602694089, 'Sulfate': None, 'Conductivity': 418.6062130644815, 'Organic_carbon': 16.868636929550973, 'Trihalomethanes': 66.42009251176368, 'Turbidity': 3.0559337496641685, 'Potability': 0}, {'index': 3, 'ph': 8.316765884214679, 'Hardness': 214.3733940856225, 'Solids': 22018.41744077529, 'Chloramines': 8.05933237743854, 'Sulfate': 356.88613564305666, 'Conductivity': 363.2665161642437, 'Organic_carbon': 18.436524495493305, 'Trihalomethanes': 100.34167436508008, 'Turbidity': 4.628770536837084, 'Potability': 0}, {'index': 4, 'ph': 9.092223456290965, 'Hardness': 181.10150923612525, 'Solids': 17978.98633892625, 'Chloramines': 6.546599974207941, 'Sulfate': 310.13573752420444, 'Conductivity': 398.4108133818447, 'Organic_carbon': 11.558279443446397, 'Trihalomethanes': 31.997992727424737, 'Turbidity': 4.075075425430034, 'Potability': 0}, {'index': 5, 'ph': 5.584086638456089, 'Hardness': 188.3133237696164, 'Solids': 28748.68773904612, 'Chloramines': 7.54486878877965, 'Sulfate': 326.6783629116736, 'Conductivity': 280.4679159334877, 'Organic_carbon': 8.399734640152758, 'Trihalomethanes': 54.91786184199447, 'Turbidity': 2.5597082275565217, 'Potability': 0}, {'index': 6, 'ph': 10.223862164528772, 'Hardness': 248.07173527013992, 'Solids': 28749.716543528237, 'Chloramines': 7.5134084658313025, 'Sulfate': 393.66339551509645, 'Conductivity': 283.6516335078445, 'Organic_carbon': 13.789695317519886, 'Trihalomethanes': 84.60355617402357, 'Turbidity': 2.672988736934779, 'Potability': 0}, {'index': 7, 'ph': 8.635848718500734, 'Hardness': 203.3615225845705, 'Solids': 13672.091763901635, 'Chloramines': 4.563008685599703, 'Sulfate': 303.3097711592812, 'Conductivity': 474.6076449424485, 'Organic_carbon': 12.36381669870525, 'Trihalomethanes': 62.79830896292516, 'Turbidity': 4.401424715445482, 'Potability': 0}  ... displaying 10 of 3276 total bound parameter sets ...  {'index': 3274, 'ph': 5.126762923351532, 'Hardness': 230.60375750846123, 'Solids': 11983.869376336364, 'Chloramines': 6.303356534249105, 'Sulfate': None, 'Conductivity': 402.883113121781, 'Organic_carbon': 11.1689462210565, 'Trihalomethanes': 77.48821310275477, 'Turbidity': 4.708658467526655, 'Potability': 1}, {'index': 3275, 'ph': 7.874671357791283, 'Hardness': 195.10229858610904, 'Solids': 17404.17706105066, 'Chloramines': 7.509305856927908, 'Sulfate': None, 'Conductivity': 327.4597604610721, 'Organic_carbon': 16.140367626166324, 'Trihalomethanes': 78.69844632549504, 'Turbidity': 2.309149056634923, 'Potability': 1})
    2022-05-07 11:43:30,065 INFO sqlalchemy.engine.Engine COMMIT
    


```python

```



