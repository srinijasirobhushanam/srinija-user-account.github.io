---
title: "Data cleaning"
date: 2018-01-28
tags: [data wrangling, data science, messy data]
header:
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
my_data = genfromtxt('Salary_Data.csv', delimiter=',',skip_header=1 )
X = np.c_[np.ones(my_data.shape[0]),my_data[:,0]]
y = np.c_[my_data[:,1]]
m = y.size 
plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'o',markersize=10)
plt.grid(True)
plt.ylabel('Salary')
plt.xlabel('Years of Experience')

def h(beta,X): #Linear hypothesis function
    return np.dot(X,beta)

def computeCost(mybeta,X,y): #Cost function
    return float((1./(2*m)) * np.dot((h(mybeta,X)-y).T,(h(mybeta,X)-y)))

initial_beta = np.zeros((X.shape[1],1))
print (computeCost(initial_beta,X,y))
iterations= 500
alpha= 0.01
#Actual gradient descent minimizing routine
def descendGradient(X, beta_start = np.zeros(2)):
    beta = beta_start
    costvec = [] #Used to plot cost as function of iteration
    betavalues = [] 
    for val in range(iterations):
        tmpbeta = beta
        costvec.append(computeCost(beta,X,y))
        betavalues.append(list(beta[:,0]))
        #Simultaneously updating theta values
        for j in range(len(tmpbeta)):
            tmpbeta[j] = beta[j] - (alpha/m)*np.sum((h(beta,X) - y)*np.array(X[:,j]).reshape(m,1))
        beta = tmpbeta
    return beta, betavalues, costvec
initial_beta = np.zeros((X.shape[1],1))
beta, betavalues, costvec = descendGradient(X,initial_beta)

def fit(xval):
    return beta[0] + beta[1]*xval
plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'o',markersize=10,label='Training Data')
plt.plot(X[:,1],fit(X[:,1]),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(beta[0],beta[1]))
plt.grid(True) #Always plot.grid true!
plt.ylabel('Years of experience')
plt.xlabel('Salary')
plt.legend()from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

fig = plt.figure(figsize=(20,20))
ax = fig.gca(projection='3d')

xvals = np.arange(-10,10,.5)
yvals = np.arange(-1,4,.1)
xs, ys, zs = [], [], []
for a in xvals:
    for b in yvals:
        xs.append(a)
        ys.append(b)
        zs.append(computeCost(np.array([[a], [b]]),X,y))

scat = ax.scatter(xs,ys,zs,c=np.abs(zs),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel(r'$\beta_0$',fontsize=30)
plt.ylabel(r'$\beta_1$',fontsize=30)
plt.title('Cost Minimization Path',fontsize=30)
plt.plot([x[0] for x in betavalues],[x[1] for x in betavalues],costvec,'bo-')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
my_data = genfromtxt('Salary_Data.csv', delimiter=',',skip_header=1 )
X = np.c_[np.ones(my_data.shape[0]),my_data[:,0]]
y = np.c_[my_data[:,1]]
m = y.size 


```python

```


```python
def h(beta,X): 
    return np.dot(X,beta)

```


```python
def computeCost(val_beta,X,y): 
    return float((1./(2*m)) * np.dot((h(val_beta,X)-y).T,(h(val_beta,X)-y)))

```


```python
initial_beta = np.zeros((X.shape[1],1))
print (computeCost(initial_beta,X,y))

```

    3251477635.366667



```python
iterations = 500
lr= 0.01

```


```python
iterations = 500
alpha = 0.01
def descendGradient(X, beta_start = np.zeros(2)):
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    """
    beta = beta_start
    xrange=range
    jvec = [] #Used to plot cost as function of iteration
    betahistory = [] #Used to visualize the minimization path later on
    for meaninglessvariable in xrange(iterations):
        tmpbeta = beta
        jvec.append(computeCost(beta,X,y))
        betahistory.append(list(beta[:,0]))
        #Simultaneously updating theta values
        for j in xrange(len(tmpbeta)):
            tmpbeta[j] = beta[j] - (alpha/m)*np.sum((h(beta,X) - y)*np.array(X[:,j]).reshape(m,1))
        beta = tmpbeta
    return beta, betahistory, jvec
```


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
```


```python
boston = load_boston()
```


```python
boston.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])




```python
boston.data.shape
```




    (506, 13)




```python
print (boston.feature_names)
```

    ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
     'B' 'LSTAT']



```python
print (boston.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    



```python

boston_dataset = pd.DataFrame(boston.data)
boston_dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python

boston_dataset.columns = boston.feature_names
boston_dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python

boston_dataset['PRICE'] = boston.target
boston_dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
boston_dataset.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns
from matplotlib import rcParams
plt.scatter(boston_dataset.CRIM, boston_dataset.PRICE)
plt.xlabel("Per capita crime rate by town")
plt.ylabel("Price of the house")
plt.title("Relationship between crime rate and Price")
```




    <matplotlib.text.Text at 0x7f0ff44efc18>




![png](output_25_1.png)



```python
plt.scatter(boston_dataset.RM, boston_dataset.PRICE)
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("Price of the house")
plt.title("Relationship between rooms per dwelling and Price")
```




    <matplotlib.text.Text at 0x7f0ff43347f0>




![png](output_26_1.png)



```python

plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-teacher ratio by town")
plt.ylabel("Price of the house")
plt.title("Relationship between PTRATIO and Price")

```




    <matplotlib.text.Text at 0x7f0ff42bbeb8>




![png](output_27_1.png)



```python

from sklearn.linear_model import LinearRegression
X = boston_dataset.RM
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, boston_dataset.PRICE, test_size=0.2, random_state = 5)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)
x_train_new= X_train.reshape(-1,1)
y_train_new= Y_train.reshape(-1,1)
x_test_new= X_test.reshape(-1,1)
y_test_new= Y_test.reshape(-1,1)


# This creates a LinearRegression object
linear_model = LinearRegression()
```

    (404,)
    (102,)
    (404,)
    (102,)


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:10: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      # Remove the CWD from sys.path while we load stuff.
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      # This is added back by InteractiveShellApp.init_path()
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      if sys.path[0] == '':
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:13: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      del sys.path[0]



```python
linear_model.fit(x_train_new,y_train_new)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python

print ('Estimated intercept coefficient:', linear_model.intercept_)
```

    Estimated intercept coefficient: 62.34462747483265



```python
pred = linear_model.predict(x_test_new)
print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - y_test_new)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - y_test_new) ** 2))
print("R2-score: %.2f" % r2_score(pred , y_test_new) )
```

    Mean absolute error: 3.47
    Residual sum of squares (MSE): 23.97
    R2-score: 0.44



```python
# multiple linear regression

X = boston_dataset.drop('PRICE', axis = 1)
```


```python
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, boston_dataset.PRICE, test_size=0.2, random_state = 5)



# This creates a LinearRegression object
linear_model = LinearRegression()
```


```python
linear_model.fit(X_train,Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
print ('Estimated intercept coefficient:', linear_model.intercept_)
```

    Estimated intercept coefficient: 37.9124870097502



```python
pred = linear_model.predict(X_test)
print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - Y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - Y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , Y_test) )
```

    Mean absolute error: 3.21
    Residual sum of squares (MSE): 20.87
    R2-score: 0.72



```python
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
my_data = genfromtxt('Salary_Data.csv', delimiter=',',skip_header=1 )
X = np.c_[np.ones(my_data.shape[0]),my_data[:,0]]
y = np.c_[my_data[:,1]]
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state = 5)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)



# This creates a LinearRegression object
linear_model = LinearRegression()
```

    (24, 2)
    (6, 2)
    (24, 1)
    (6, 1)



```python
linear_model.fit(X_train,Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
print ('Estimated intercept coefficient:', linear_model.intercept_)
```

    Estimated intercept coefficient: [26065.29298314]



```python
pred = linear_model.predict(X_test)
print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - Y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - Y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , Y_test) )
```

    Mean absolute error: 4641.30
    Residual sum of squares (MSE): 35369798.22
    R2-score: 0.94



```python

```
