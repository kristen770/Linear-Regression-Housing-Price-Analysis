#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


#library imports 
import pandas as pd  
import pandasql
import numpy as np  

from statsmodels.formula.api import ols 
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import statsmodels.api as sm
import scipy.stats as stats 
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from itertools import combinations

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  

import warnings
warnings.filterwarnings('ignore') 


# ## Data Initialization and Visualization
# Import the data and visualize 

# In[2]:


get_ipython().run_line_magic('store', '-r ols_df')
ols_df.head()


# In[3]:


#copy working df to a df called: lr_df
ols_df = ols_df.drop(['id'], axis=1)


# In[4]:


#data distribution visualization  
pd.plotting.scatter_matrix(ols_df, alpha=0.2, figsize=(20,18))


# ## Functions

# In[5]:


def correlation_check(x_cols, df, outcome='price'):  
    """outcome= taget column name
        x_cols - independent variables 
        df = working dataframe 
        
        Returns:
        correlation 
        heatmap 
        """
    #correlation check
    feats = x_cols 
    corr = df[feats].corr()  
    
    #heatmap 
    sns.heatmap(corr, center=0, annot=True)   
     
    print("Correlation:", corr)


# In[6]:


def linear_regression(x_cols, df, outcome='price'):  
    """outcome= taget column name
        x_cols - independent variables 
        df = working dataframe 
        
        returns 
        baseline R2 
        OLS Model Summary 
        VIF scores 
        QQPlot 
        Pvalues Table"""

    #baseline model 
    y = df[['price']]
    X = df.drop(['price'], axis=1) 

    regression = LinearRegression()
    crossvalidation = KFold(n_splits=3, shuffle=True, random_state=1) 
    
    baseline= np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))
    baseline 
    
    #fit model 
    predictors = '+'.join(x_cols) 
    formula = outcome + "~" + predictors
    model = ols(formula=formula, data=df).fit()
    model.summary() 
    
    #vif scores
    X = df[x_cols] 
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]  
    
    #normality check
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)  
    
    #pvalues
    summary = model.summary()
    p_table = summary.tables[1]
    p_table = pd.DataFrame(p_table.data)
    p_table.columns = p_table.iloc[0]
    p_table = p_table.drop(0)
    p_table = p_table.set_index(p_table.columns[0])
    p_table['P>|t|'] = p_table['P>|t|'].astype(float)
    x_cols = list(p_table[p_table['P>|t|'] < 0.05].index)
    x_cols.remove('Intercept')
    
    print("Baseline:", baseline)  
    print(model.summary()) 
    print("\n")
    print("Vif Scores:", list(zip(x_cols, vif)))   
    print("\n")
    print(len(p_table), len(x_cols))
    print(x_cols[:5])
    print(p_table.head())


# In[7]:


def build_sm_ols(df, features_to_use, target, add_constant=False, show_summary=True):
    X = df[features_to_use]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X).fit()
    if show_summary:
        print(ols.summary())
    return ols

# assumptions of ols
# residuals are normally distributed
def check_residuals_normal(ols):
    residuals = ols.resid
    t, p = stats.shapiro(residuals)
    if p <= 0.05:
        return False
    return True


# residuals are homoskedasticitous
def check_residuals_homoskedasticity(ols):
    import statsmodels.stats.api as sms
    resid = ols.resid
    exog = ols.model.exog
    lg, p, f, fp = sms.het_breuschpagan(resid=resid, exog_het=exog)
    if p >= 0.05:
        return True
    return False




def check_vif(df, features_to_use, target_feature):
    ols = build_sm_ols(df=df, features_to_use=features_to_use, target=target_feature, show_summary=False)
    r2 = ols.rsquared
    return 1 / (1 - r2)
    
    
    
# no multicollinearity in our feature space
def check_vif_feature_space(df, features_to_use, vif_threshold=3.0):
    all_good_vif = True
    for feature in features_to_use:
        target_feature = feature
        _features_to_use = [f for f in features_to_use if f!=target_feature]
        vif = check_vif(df=df, features_to_use=_features_to_use, target_feature=target_feature)
        if vif >= vif_threshold:
            print(f"{target_feature} surpassed threshold with vif={vif}")
            all_good_vif = False
    return all_good_vif
        
        


def check_model(df, 
                features_to_use, 
                target_col, 
                add_constant=False, 
                show_summary=False, 
                vif_threshold=3.0):
    has_multicollinearity = check_vif_feature_space(df=df, 
                                                    features_to_use=features_to_use, 
                                                    vif_threshold=vif_threshold)
    if not has_multicollinearity:
        print("Model contains multicollinear features")
    
    # build model 
    ols = build_sm_ols(df=df, features_to_use=features_to_use, 
                       target=target_col, add_constant=add_constant, 
                       show_summary=show_summary)
    
    # check residuals
    resids_are_norm = check_residuals_normal(ols)
    resids_are_homo = check_residuals_homoskedasticity(ols)
    
    if not resids_are_norm or not resids_are_homo:
        print("Residuals failed test/tests")
    return ols


# In[8]:


def corr_function(x, y):
    try:
        return x.corr(y)
    except:
        return None


# In[10]:


outcome = "price"


# ## Identifying Correlated Features

# In[16]:


# seperate data by correlation values

columns_correlations = []
columns_non_numeric = []

for column in ols_df.drop(columns=[outcome]).columns:
    try:
        corr = np.abs(ols_df[column].corr(ols_df[outcome]))
        t = (column, corr)
        columns_correlations.append(t)
    except:
        columns_non_numeric.append(column) 
columns_correlations


# In[17]:


correlated_features_above_2 = [t[0] for t in columns_correlations if t[1] >= 0.20]
correlated_features_above_2


# In[20]:


correlated = correlated_features_above_2  
correlated.append(outcome)
trial1 =ols_df[correlated]


# In[21]:


trial1.head()


# In[22]:


pd.plotting.scatter_matrix(trial1, figsize=(20, 20))
plt.show()


# ## Trial 1 
# All columns that were correlated

# In[25]:


trial1.head()


# In[26]:


#redefine the problem 
outcome = 'price'
t1 = ['sqft_living', 'floors', 'grade', 'sqft_above', 'lat', 'sqft_living15']


# In[27]:


linear_regression(x_cols=t1, df=trial1, outcome='price')


# ## Trial 2 
# Remove 'sqft_basement' it has a vif score of 670 indicating high multicollinearity

# In[32]:


trial1.head()


# In[29]:


outcome = 'price'
t2 = ['sqft_living', 'floors', 'grade', 'sqft_above', 'lat', 'sqft_living15']


# In[34]:


linear_regression(x_cols=t2, df=trial1, outcome='price')


# In[ ]:




