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

# In[2]:


get_ipython().run_line_magic('store', '-r df')
df.head()


# In[3]:


#copy working df to a df called: lr_df
lr_df = df.copy() 
lr_df = lr_df.drop(['date', 'id'], axis=1)


# In[4]:


#data distribution visualization  
pd.plotting.scatter_matrix(lr_df, alpha=0.2, figsize=(20,18))


# ## Functions

# In[38]:


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


def calc_slope(xs, ys):

    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)**2) - np.mean(xs*xs)))
    
    return m


# ## Create Test and Trial Data  
# Sepearte the data so that we can train and test on different values.

# In[8]:


#train & test groups
train, test = train_test_split(lr_df) 
print("Train:", len(train), "Test:", len(test))


# In[9]:


train.head()


# ## Trial 1 
# This trial is done with the full data set, unmanipulated 

# In[10]:


#define the problem 
outcome = 'price'
t1 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']


# In[11]:


linear_regression(x_cols=t1, df=train, outcome='price')


# In[13]:


#correlation test 
correlation_check(x_cols=t1, df=test, outcome='price')


# The rsquared value for this trial (0.692) is relatively low and the qq plot shows there is a large tail my refining will be to drop outlying data from the train data.

# ## Trail 2 
# Remove outlying data greater than 3 STD from the mean

# In[14]:


#define the problem 
outcome = 'price'
t2 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']


# In[15]:


#look at z scores (relationship to the std & mean) 
z = np.abs(stats.zscore(train)) 
threshold = 3 #3 std away from them mean
print(np.where(z > 3))


# In[16]:


#drop anything outside 3 std and set it to a new dataframe named "df_1"
train = train[(z<3).all(axis=1)]
train.head() 


# In[17]:


linear_regression(x_cols=t2, df=train, outcome='price')


# Removing the outliers dropped the rsquared value (0.675) but created a less exagerated tail on the qqplot. My next refinment will be to drop any columns whose PValue is > 0.05. 

# ## Trial 3 
# Removing "sqft_lot" & "sqft_basement" do to higher than 0.05 pvalues

# In[18]:


#redefine the problem 
outcome = 'price'
t3 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']


# In[19]:


linear_regression(x_cols=t3, df=train, outcome='price')


# Removing the columns did not have an affect on the rsquared score of the model. The next piece of model refinement will be to remove vif scores greater than 100.

# ## Trial 4 
# Remove 'yr_built', 'lat', 'long', 'grade'

# In[20]:


#redefine the problem 
outcome = 'price'
t4 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition',
          'sqft_above', 'yr_renovated', 'sqft_living15', 'sqft_lot15']


# In[21]:


linear_regression(x_cols=t4, df=train, outcome='price')


# In[22]:


correlation_check(x_cols=t4, df=train, outcome='price')


# Removing these columns had a detrimental affect on my rsquared score (dropping it to 0.448) the warnings indicate that there might be strong collinearity. Preforming a colleniarity check shows that sqft_living & sqft_above have a 0.86 correlation and sqft_living & sqft_living15 have a 0.75 correlation so I will remove those two from the next trial

# ## Trial 5
# Remove sqft_living15 & sqft_above 

# In[23]:


#redefine the problem 
outcome = 'price'
t5 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'yr_renovated', 'sqft_lot15']


# In[24]:


linear_regression(x_cols=t5, df=train, outcome='price')


# In[25]:


correlation_check(x_cols=t5, df=train, outcome='price')


# This adjustment worsened the rsquared score. Looking at a correlation heatmap it looks like there is still strong correlation between bedrooms & sqft_living and bathrooms & sqft_living. I am going to remove sqft living from the next trial. 

# # Trial 6 
# Remove sqft_living

# In[26]:


#redefine the problem 
outcome = 'price'
t6 = ['bedrooms', 'bathrooms', 'floors', 'condition', 'yr_renovated', 'sqft_lot15']


# In[27]:


linear_regression(x_cols=t6, df=train, outcome='price')


# Ultimately this is not producing good results and my rquared score is dropping. I am going to try a new approach to selecting features in a new notebook "Linear Regression Trails 2"

# In[39]:


ols = check_model(df=train, features_to_use=t6, 
                     target_col='price', show_summary=True)

