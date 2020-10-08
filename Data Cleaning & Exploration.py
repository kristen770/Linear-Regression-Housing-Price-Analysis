#!/usr/bin/env python
# coding: utf-8

# ## Final Project Submission
# 
# Please fill out:
# * Student name: Kristen Davis 
# * Student pace: Full Time
# * Scheduled project review date/time: Monday October 19 - 11:30 am
# * Instructor name: Rafael Carrasco
# * Blog post URL:
# 

# # To Do:  
# 2. look up king count "grading system" 
# 3. what constitutes a 1 bathroom a .5 and a .25 bathroom 
# 4. convert date into data time column 
# 5. make df by zipcodes 
# 6. deal with ? in the sft_basement 
# 
# # Check List: 
# 1. Stateholders defined
# 2. Buisness problem defined  
# 3. 3 Posed Questions
# 3. All functions need docstrings 
# 4. A webscraped data set 
# 4. 4 data visualizations  
# 5. 1-2 intro notebook paragraphs
# 6. no p-values higher than 0.05 
# 7. an iterative approach to modeling
# 8. explain in a paragraph every iterative change 
# 9. 1 parapgraph explaining your final model 
# 10. at least 3 coefficients and explain thier inpact on the model
# 11. readme 
# 12. slide deck 
# 13. recording 
# 
# # Helpful Sites 
# https://towardsdatascience.com/worthless-in-seattle-9300b3594383 
# https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b 
# https://www.kingcounty.gov/depts/community-human-services/housing/services/housing-repair.aspx 
# http://www.socialserve.com/tenant/WA/index.html?ch=KCHA
# http://seattlecitygis.maps.arcgis.com/apps/webappviewer/index.html?id=f822b2c6498c4163b0cf908e2241e9c2 
# https://www.kaggle.com/harlfoxem/housesalesprediction/notebooks 
# https://github.com/EricaSG/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320 
# https://github.com/saifzkb/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320 
# https://github.com/jari-el13/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320/blob/master/student.ipynb 
# https://github.com/kailakay/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320/blob/master/Q1.ipynb
# https://github.com/mclaurenhr/Mod02-Linear-Regression

# In[ ]:


#working dataframes 

raw_data_df #the inital dataframe that was loaded from the csv 
df #primary working df (feature engineered & cleaned data )
burbs #zipcodes outside the seattle city limits 
seattle_proper #zipcode inside the seattle city limits 
pop_density_by_zip #zip code and its population density 
low_density #the zipcode with less than 2,500 pop density 
med_density #the zipcode with between 2,500 and 5,000 pop density 
high_density #the zipcode with higher than 5,000 pop density  
inital_df #df I used on inital model  
df_full #all columns linear regression working dataframe


# # Data Cleaning 

# ## Workspace Set Up

# In[298]:


#library imports 
import pandas as pd  
#pip install pandasql
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
import plotly.express as px

import warnings
warnings.filterwarnings('ignore') 


# In[189]:


#Read in housing data & initialize it to a variable
raw_data_df = pd.read_csv("data/kc_house_data.csv") 


# ## Data Cleaning 

# In[176]:


#explore inital dataset 
raw_data_df.info()   
# There are 21,597 Observations (each row is a house sale)
# Waterfront, view, year renovated have null values   
# Date is object  
# There are "?" in the 

len(raw_data_df[raw_data_df.duplicated()]) 
#0 all unique home sales 


# In[177]:


#looking at the nan values 
raw_data_df.isna().sum() 
#watefront 2,376/ 21,597 NAN 
#view 63/ 21,597 NAN 
#yr_renovated 3,842/21,597 NAN  


# In[449]:


#replace NAN values with 0 
df = raw_data_df.fillna(0) 
df 

#veriying NAN values are gone
df.isna().sum()


# In[179]:


#change the date row to a datetime object 
df['date'] = pd.to_datetime(df['date'])


# In[180]:


#looking at values in sqft_basement 
df['sqft_basement'].unique()
df = df.replace({'sqft_basement': {"?": 0.0}}) 

#set as float 
df['sqft_basement'] = df['sqft_basement'].astype('float64')


# ##### Data Decision Explaination: 
# There are too many rows with NAN (particularly in the waterfront column) to drop without affecting the dataset, given that if all three of these coloumns are indicators that not every house would have, and opperating under the assumption that if the house did have any one of there three 'features' homeowners would be motivated to list them - I am going to replace all NaN with a 0 to indicate that the house does not have that feature.

# In[181]:


get_ipython().run_line_magic('store', 'df')


# In[190]:


df.columns


# ## Feature Engineering  
# Additional features added to the data frame include: population density column 

# ### Creating a Population Density Feature
# Based on the information gathered <a href="https://www.unitedstateszipcodes.org/98039/">here</a> I created a column that lists the population per square mile in 2015 by zipcode. This will be helpful in considering a model that is senstive to urban v rural areas.

# In[200]:


#creating a dataframe of each zipcode & population density data from 2015 called pop_density_by_zip
zipcode = [98178, 98125, 98028, 98136, 98074, 98053, 98003, 98198, 98146,
       98038, 98007, 98115, 98107, 98126, 98019, 98103, 98002, 98133,
       98040, 98092, 98030, 98119, 98112, 98052, 98027, 98117, 98058,
       98001, 98056, 98166, 98023, 98070, 98148, 98105, 98042, 98008,
       98059, 98122, 98144, 98004, 98005, 98034, 98075, 98116, 98010,
       98118, 98199, 98032, 98045, 98102, 98077, 98108, 98168, 98177,
       98065, 98029, 98006, 98109, 98022, 98033, 98155, 98024, 98011,
       98031, 98106, 98072, 98188, 98014, 98055, 98039]
pop_density = [4966, 6879, 3606, 6425, 2411, 662, 3800, 4441, 5573, 469, 5684, 7018, 9602, 6732, 
               141, 9905, 4423, 6279, 3591, 892, 4741, 8638, 6667, 2908, 469, 7953, 2215, 1717, 4323, 
               3580, 4604, 288, 3194, 10643, 1537, 4437, 1725, 13594, 7895, 3977, 2361, 4428, 2185, 
               7523, 334, 6841, 4714, 2024, 41, 15829, 785, 2989, 3794, 3341, 171, 2719, 3402, 10361, 52,
               3696, 4330, 236, 3569, 4877, 4161, 1231, 3062, 149, 4585, 2059] 

lists = list(zip(zipcode, pop_density)) 
pop_density_by_zip = pd.DataFrame(lists, columns = ['zipcode', 'pop_density'])


# In[202]:


#map population density onto the main dataframe as a new column called "pop_density"
dic = {zipcode[i]: pop_density[i] for i in range(len(zipcode))}  

def set_value(row_number, assigned_value): 
    return assigned_value[row_number] 

event_dictionary = {98178: 4966, 98125: 6879, 98028: 3606, 98136: 6425, 98074: 2411, 98053: 662, 98003: 3800, 98198: 4441, 
                    98146: 5573, 98038: 469, 98007: 5684, 98115: 7018, 98107: 9602, 98126: 6732, 98019: 141, 98103: 9905,
                    98002: 4423, 98133: 6279, 98040: 3591, 98092: 892, 98030: 4741, 98119: 8638, 98112: 6667, 98052: 2908, 
                    98027: 469, 98117: 7953, 98058: 2215, 98001: 1717, 98056: 4323, 98166: 3580, 98023: 4604, 98070: 288, 
                    98148: 3194, 98105: 10643, 98042: 1537, 98008: 4437, 98059: 1725, 98122: 13594, 98144: 7895, 98004: 3977,
                    98005: 2361, 98034: 4428, 98075: 2185, 98116: 7523, 98010: 334, 98118: 6841, 98199: 4714, 98032: 2024, 
                    98045: 41, 98102: 15829, 98077: 785, 98108: 2989, 98168: 3794, 98177: 3341, 98065: 171, 98029: 2719, 
                    98006: 3402, 98109: 10361, 98022: 52, 98033: 3696, 98155: 4330, 98024: 236, 98011: 3569, 98031: 4877, 
                    98106: 4161, 98072: 1231, 98188: 3062, 98014: 149, 98055: 4585, 98039: 2059} 
df['pop_density'] = df['zipcode'].apply(set_value, args=(event_dictionary, )) 


# In[203]:


#subdivide my dataframe by density into roughly equal sections 
low_density = df.loc[df['pop_density'] < 2500] 
med_density = df.loc[(df['pop_density'] > 2500) & (df['pop_density'] < 5000)] 
high_density = df.loc[df['pop_density'] > 5000]


# # Categorical Data  
# The following columns have data that I am going to treat as categorical: "zipcode", "yr_bui't" & "yr_renovated"

# In[214]:


#create plot to visualize data types
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(16,3)) 

all_cols =['pop_density', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
           'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
           'sqft_living15', 'sqft_lot15']

for xcol, ax in zip(['pop_density', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode'], axes):
    df.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')


# In[216]:


#histogram of the columns 
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
df.hist(ax = ax);


# In[434]:


lr_df = df.copy() 
get_ipython().run_line_magic('store', 'lr_df')


# In[435]:


lr_df.columns


# In[436]:


df.columns


# ## Label Encode "yr_built" 
# I am going to encode the yr_built column into three values 1900-1950, 1950-2000, 2000-2015 this will hopefully add more meaning to the yr_built column in the ols model?

# In[427]:


#create dummy categories 
yrs = ['Before 1950', 'Between 1950-2000', 'After 2000']
yrs_series = pd.Series(yrs)


# In[428]:


#set values as string
yrs_origin = yrs_series.astype('category')
yrs_origin


# In[429]:


#assign to dataframe 
lr_df["yr_built"] = lr_df["yr_built"].map(lambda x: '0' if x < 1950 else ('2' if x > 2000  else '1'))


# ## Dummify "yr_renovated" 
# I am going to dummify this to either it has been renovated or it hasn't. This should help account for the vast majority of 0 values in the columnm. Only 744 of the houses had a value for this which is only 3% of the data set.

# In[450]:


df.loc[df['yr_renovated'] != 0]


# In[442]:


#map a 1 if renovated 0 if not 
lr_df['yr_renovated'] = lr_df['yr_renovated'].map(lambda x: '1' if x < 1 else '0')


# In[443]:


#check data has been dummified 
lr_df['yr_renovated'].unique()


# ## Dummify "Zipcode" 
# Dummify the zipcode columns into groupings that are inside and outside of the city limits I found the city border
# <a href="https://www.usmapguide.com/washington/seattle-zip-code-map">here</a> this will allow me to group the zipcodes without splitting up my data too much 

# In[452]:


#unique zipcodes 
df['zipcode'].unique() 


# In[453]:


#inside city line
options = [98155, 98177, 98133, 98125, 98117, 98103, 98115, 98105, 98102, 98112, 98109, 98107, 98119, 
           98199, 98122, 98144, 98134, 98108, 98118, 98168, 98106, 98126, 98136, 98116, 98146, 98178, 98121,
           98101, 98154, 98104] 
seattle_proper = df[df['zipcode'].isin(options)] 


# In[454]:


#breaking up zipcode by seattle city limits  
#outside city line
non_city_options = [98028, 98074, 98053, 98003, 98198, 98038, 98007, 98019, 98002, 98040, 98092, 98030, 98052, 98027, 98058, 
98001, 98056, 98166, 98023, 98070, 98148, 98042, 98008, 98059, 98004, 98005, 98034, 98075, 98010, 98032, 98045, 98077,  
98065, 98029, 98006, 98022, 98033, 98024, 98011, 98031, 98072, 98188, 98014, 98055, 98039]
burbs = df[df['zipcode'].isin(non_city_options)]   


# In[460]:


dic = {zipcode[i]: pop_density[i] for i in range(len(zipcode))}  

def set_value(row_number, assigned_value): 
    return assigned_value[row_number] 

event_dictionary = {98155: 'city', 98177: 'city', 98133: 'city', 98125: 'city', 98117: 'city', 98103: 'city', 98115: 'city',
                    98105: 'city', 98102: 'city', 98112: 'city', 98109: 'city', 98107: 'city', 98119: 'city', 98199: 'city',
                    98122: 'city', 98144: 'city', 98134: 'city', 98108: 'city', 98118: 'city', 98168: 'city', 98106: 'city',
                    98126: 'city', 98136: 'city', 98116: 'city', 98146: 'city', 98178: 'city', 98121: 'city', 98101: 'city',
                    98154: 'city', 98104: 'city', 98028: 'notcity', 98074: 'notcity', 98053: 'notcity', 98003: 'notcity', 
                    98198: 'notcity', 98038: 'notcity', 98007: 'notcity', 98019: 'notcity', 98002: 'notcity', 98040: 'notcity',
                    98092: 'notcity', 98030: 'notcity', 98052: 'notcity', 98027: 'notcity', 98058: 'notcity', 98001: 'notcity', 
                    98056: 'notcity', 98166: 'notcity', 98023: 'notcity', 98070: 'notcity', 98148: 'notcity', 98042: 'notcity', 
                    98008: 'notcity', 98059: 'notcity', 98004: 'notcity', 98005: 'notcity', 98034: 'notcity', 98075: 'notcity', 
                    98010: 'notcity', 98032: 'notcity', 98045: 'notcity', 98077: 'notcity', 98065: 'notcity', 98029: 'notcity',
                    98006: 'notcity', 98022: 'notcity', 98033: 'notcity', 98024: 'notcity', 98011: 'notcity', 98031: 'notcity', 
                    98072: 'notcity', 98188: 'notcity', 98014: 'notcity', 98055: 'notcity', 98039: 'notcity'} 
lr_df['city'] = lr_df['zipcode'].apply(set_value, args=(event_dictionary, )) 


# In[461]:


#set city to 0 and notcity to 1 
lr_df['city'] = lr_df['city'].map(lambda x: '0' if x == "city" else '1')


# In[462]:


#check to make sure dummy updated 
lr_df['city'].unique()


# ## Encode "grade" 
# I am going to encode grade because it is on a scale currently from 3-13 so encoding it will reset the represented value scale

# In[468]:


#labels 
grade = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10}
lr_df.replace(grade, inplace=True)
lr_df['grade'].unique()


# # Remove Outliers  
# Remove any values 3 standard deviations above the mean 

# In[ ]:


#drop date column 
lr_df = lr_df.drop(['date'], axis=1)


# In[498]:


lr_df.dtypes


# ##### Data Decision Analysis 
# Assuming that the definiton of a full or "1.0" bathroom is that is has a sink, toliet, and shower (3 appliances) I am assuming that each of these whole bathrooms is correlated to a bedroom. In order to reduce the multicolinearity of these two columns I am going to leave only the .50 (just toliet and sink) and .25 (just sink) values in the column to indicate a value of bathrooms that are in addition to the bedroom bathroom.

# In[503]:


#resent dtype on encoded columns
lr_df['yr_renovated'] = lr_df['yr_renovated'].astype('float64') 
lr_df['city'] = lr_df['city'].astype('int64')


# In[504]:


#look at z scores (relationship to the std & mean) 
z = np.abs(stats.zscore(lr_df)) 
threshold = 3 #3 std away from them mean
print(np.where(z > 3))


# In[505]:


#drop anything outside 3 std and set it to a new dataframe named "df_1"
lr_df = lr_df[(z<3).all(axis=1)]
lr_df.head() 


# # Bedrooms & Bathrooms 
# Modify the relationship between bedrooms and bathrooms to reduce multicolinearity

# In[484]:


#High skew in the data of bedrooms when investigated there is a 33 bedroom house
lr_df['bedrooms'].unique()
#the 33 bedrooms in this seem to be a mistake I am going to drop the row 


# In[483]:


#lr_df.drop([15856], inplace=True)


# In[490]:


lr_df['bathrooms'].unique() 
lr_df.loc[lr_df['bathrooms'] == 7.75]


# In[521]:


lr_df['bathrooms'].unique()


# In[525]:


lr_df['sqft_lot'].unique()


# # Log Transform 
# Transform non normal distributions in the data

# In[506]:


lr_df.columns


# In[522]:


#visulaize distributions  
x_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',]
pd.plotting.scatter_matrix(lr_df[x_cols], figsize=(10,12));


# In[527]:


#visulaize distributions  
x_cols3 = ['sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'pop_density']
pd.plotting.scatter_matrix(lr_df[x_cols3], figsize=(10,12));


# In[531]:


#look just at non normal 
non_normal = ['sqft_lot', 'sqft_lot15', 'sqft_living15']
for feat in non_normal:
    lr_df[feat] = lr_df[feat].map(lambda x: np.log(x))
pd.plotting.scatter_matrix(lr_df[non_normal], figsize=(10,12));


# In[533]:


ols_df = lr_df.copy()  
get_ipython().run_line_magic('store', 'ols_df')


#  # EDA 

# ## What was the average amount spent on a house by a buyer?

# In[10]:


#what is the median price of all houses?
df['price'].mean() 
#$540,296.57

#what is the median price of all houses? 
df['price'].median()  
#$ 450,000.00


# ## When were the houses in this data set built? 

# In[301]:


#Isolate the year built feature and count the number of occurances in the dataframe
df1 = df.groupby('yr_built').count()    
df1 = df1.drop(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
       'grade', 'sqft_above', 'sqft_basement', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
        'pop_density'], axis=1) 
df1.reset_index(inplace=True)


# In[302]:


#Add a percentage of of the whole data frame column to look at when the majority of houses were built 
df1['percentage_whole'] = round(((df1.id / 21597) * 100), 2)


# In[303]:


#Create a list of each year and the number of houses in the dataframe that were built in that year
year_quantity = df1.to_records(index=False)
result = list(year_quantity) 
result.sort(key = lambda x: x[1])
print(result)


# In[305]:


#Finding the total number of houses that have been built in each year and the percentage of total homes that represents
df1.loc[df1["yr_built"] > 2000] 


# In[306]:


df1.loc[df1["yr_built"] > 2000].sum() 


# In[307]:


df1.loc[df1["percentage_whole"] > 2]


# #### Data Observation 
# 21% of the houses in this dateframe have been built after 2000 - there was a sharp increase in the number of homes that were built between 2013 and 2014 - between 2011 and 2016 Amazon experienced a 500% increase in it's number of employees which probably contributed to the number of houses being newly built and sold in those years. The largest percentages of this data are represented in 2005(2.08), 2006(2.10) and 2014(2.59) <a href="https://en.wikipedia.org/wiki/History_of_Amazon">source</a>

# In[309]:


#all years / percentage of whole 
fig = px.line(df1, x="yr_built", y="percentage_whole", title='Houses Sold By Year Built')
fig.show()


# In[310]:


#all years / percentage of whole line 
fig = px.line(df1, x="yr_built", y="id", title='Houses Sold By Year Built')
fig.show()


# In[399]:


#all years/ number of houses bar
fig = px.bar(df1, y='id', x='yr_built')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', marker_color='#7b9e86', opacity=0.75)
fig.update_layout( 
    uniformtext_minsize=8, uniformtext_mode='hide',
    title={
        'text': "Houses Sold By Year Build", 
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
    
    xaxis=dict(
        title='Year Built',
        titlefont_size=18,
        tickfont_size=14, 
    ), 
    yaxis=dict(
        title='Number of Houses',
        titlefont_size=18,
        tickfont_size=14,
    ),
    
    #paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
   
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1, # gap between bars of the same location coordinate. 
) 

fig.show()


# ## What is the distribution of housing grade in the data?

# In[17]:


#Identifty the amount of each 'quality' of house in the data
df.loc[df['grade'] == 13] 
# None: 0, 1, 2 
# 1: 3 "Falls short of minimum building standards. Normally cabin or inferior structure."
# 27: 4 "Generally older, low quality construction. Does not meet code" 
# 242: 5 "Low construction costs and workmanship. Small, simple design." 
# 2038: 6 "Lowest grade currently meeting building code. Low quality materials and simple designs." 
# 8974: 7 "Average grade of construction and design. Commonly seen in plats and older sub-divisions."
# 6065: 8 "Just above average in construction and design.Usually better materials in both the exterior and interior finish work."
# 2616: 9 "Better architectural design with extra interior and exterior design and quality."
# 1134: 10 "Homes of this quality generally have high quality features. Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage."
# 399: 11 "Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options."
# 89: 12 "Custom design and excellent builders. All materials are of the highest quality and all conveniences are present."
# 13: 13 "Generally custom designed and built. Mansion level. Large amount of highest quality cabinet work, wood trim, marble, entry ways etc."
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
Y = [0, 0, 0, 1, 27, 242, 2038, 8974, 6065, 2616, 1134, 399, 89, 13]

#Simple bar chart visualization of the housing grade/ number of houses 
plt.bar(X, Y) 
plt.title("Number of House Per Grading Code in Kings County") 
plt.xlabel("Housing Grade") 
plt.ylabel("Number of Houses")  
plt.show() 

# The majoritiy of houses in this data set are in the "average" grade range (7/8) there is a larger distribution of houses 
# on the high end of the grading scale (9-13) than on the lower end (0-6). houses with a grade of 3 or 13 could be identified
# as outliers.


# In[18]:


#Investigating the zipcodes represented in each of the grading areas 
df['zipcode'].drop_duplicates() 
#There are 70 zipcodes in this data set
zips_explore = df.loc[df['grade'] == 9]['zipcode'] 
len(zips_explore.unique()) 
#49 zipcodes have grade 5 houses in them / 68 zipodes have grade 9 houses in them 
# Based this it seems like there is a good distribution of grade by zipcode  

#The total number of zipcode represented at each of the grading scales 
zips_grade = df[['grade', 'zipcode']].copy() 
zips_grade_count = zips_grade.drop_duplicates().groupby('grade').count() 
zips_grade_count = zips_grade_count.reset_index()
sns.barplot(x="grade", y="zipcode", data=zips_grade_count).set_title("Number of Zipcodes with Homes In Each Housing Grade")

#This indicated that while there is a slight skew of higher grade houses overall you can find houses grade 6- 11 at over 
# 50 of the county's 70 zipcodes 


# In[19]:


#look at values a house condition can be rated as 
df['condition'].unique() 
# Condition Grades: 1, 2, 3, 4, 5    

df.loc[df['condition'] ==5 ]
# 29 -1 Poor- Worn out. Repair and overhaul needed on painted surfaces, roofing, plumbing, heating and numerous functional inadequacies. Excessive deferred maintenance and abuse, limited value-in-use, approaching abandonment or major reconstruction; reuse or change in occupancy is imminent. Effective age is near the end of the scale regardless of the actual chronological age.
# 170 - 2 Fair- Badly worn. Much repair needed. Many items need refinishing or overhauling, deferred maintenance obvious, inadequate building utility and systems all shortening the life expectancy and increasing the effective age.
# 14020 -3 Average- Some evidence of deferred maintenance and normal obsolescence with age in that a few minor repairs are needed, along with some refinishing. All major components still functional and contributing toward an extended life expectancy. Effective age and utility is standard for like properties of its class and usage.
# 5677 - 4 Good- No obvious maintenance required but neither is everything new. Appearance and utility are above the standard and the overall effective age will be lower than the typical property.
# 1701 -5 Very Good- All items well maintained, many having been overhauled and repaired as they have shown signs of wear, increasing the life expectancy and lowering the effective age with little deterioration or obsolescence evident with a high degree of utility.
X = [1, 2, 3, 4, 5]
Y = [29, 170, 14020, 5677, 1701]

#Simple bar chart visualization of the housing grade/ number of houses 
plt.bar(X, Y) 
plt.title("Number of House Per Condition Grade in Kings County") 
plt.xlabel("Condition Grade") 
plt.ylabel("Number of Houses")  
plt.show() 


# In[20]:


#Houses that are below the condition grade and housing grade
low_grade_condition = df.loc[(df['condition'] < 3) & (df['grade'] <= 5)] 
low_grade_condition.groupby('zipcode').count() 
#There are only 1 or 2 houses that fall below the condition and grade minimum in the city in 16 of the cities 72 zipcode 
# and 1 zipcode with 7 houses 98168


# ##### Data Decision Explanation: 
# There is a relatively high distribution of houses in the 98168 zipcode that are in both poor condition and recieved a low housing grade. I chose to look more closely at the overall make up of the houses listed in this zipcode.

# In[21]:


low_grade_condition.loc[low_grade_condition['zipcode'] == 98168]['price'].mean() 
#The median price of these homes is 117,071.43 


# In[22]:


#investigating zipcode 98168
len(df.loc[df['zipcode'] == 98168]) 
#there are 269 houses listed in this zipcode 

df.loc[df['zipcode'] == 98168]['grade'].mean() 
#The mean housing grade in this neighborhood is 6.5 

df.loc[df['zipcode'] == 98168]['condition'].mean()  
#The mean condition grade in this neighborhoos is 3.2 

df.loc[df['zipcode'] == 98168]['price'].mean() 
#The mean price is 240,328.37 

#This means that the 7 houses that are in poor condition are at high likelyhood of being bought/flipped/ gentrified and would 
#be good intervetion renovation spots for the city


# In[23]:


#zipcode 98168 data trends 
z98168 = df.loc[df['zipcode'] == 98168]  

#years houses were built 
z98168['yr_built'].unique()    
len('yr_built') 
#71 years built  

z98168_after2000 = z98168.loc[z98168['yr_built'] > 2000] 
z98168_before2000 = z98168.loc[z98168['yr_built'] < 2000] 
z98168_before2000['price'].mean()  
#Mean price of homes built before 2000 233898.16 
z98168_after2000['price'].mean()  
#Mean price of homes built after 200 357450.00


# ### Housing Price by Zipcode

# In[24]:


#price_by_zips = df.loc 
df.groupby('zipcode').mean()


# In[25]:


#swarmplot of grade by zipcode 
ax = sns.boxplot(x="zipcode", y='grade', data=zips_grade)
ax = sns.swarmplot(x='zipcode', y='grade', data=zips_grade, color = ".25")


# ## Question 2: What zipcodes are being renovated at the highest rates? 

# ## Question 3: AB Testing of House Features 

# ###### Data Decision Analysis 
# Dropping the outling data that was higher than 3 standard deviations from the mean did loose alomst 13% of the data. This is not an insignifigant amount this data contained a large number of particularly expensive houses which was giving the data a heavy right tail. By dropping these we are model will become less acurate in the highest and lowest ends of the market.

# ## Trial 1: Full Data Set

# In[87]:


#Finding a cutoff point
for i in range(90, 99):
    q = i / 100
    print('{} percentile: {}'.format(q, full_cleaned['price'].quantile(q=q)))


# In[91]:


subset = full_cleaned[full_cleaned['price'] < 950000]
print('Percent removed:',(len(full_cleaned) - len(subset))/len(full_cleaned)) 


# In[ ]:


subset.head()


# In[80]:


subset.columns


# In[ ]:


test.head()


# In[107]:


y = full_cleaned[['price']]
X = full_cleaned.drop(['price'], axis=1)


# #### Fitting The Inital Model

# In[105]:


#homoskadaticy check 
plt.scatter(model.predict(train[x_cols]), model.resid)
plt.plot(model.predict(train[x_cols]), [0 for i in range(len(train))])


# #### What is the interaction between price and ? LOOK AT AN INTERACTION

# #### LOOK AT A POLYNOMIAL REGRESSION

# ## Other Garbage

# In[ ]:


# Extract the p-value table from the summary and use it to subset our features
summary = model.summary()
p_table = summary.tables[1]
p_table = pd.DataFrame(p_table.data)
p_table.columns = p_table.iloc[0]
p_table = p_table.drop(0)
p_table = p_table.set_index(p_table.columns[0])
p_table['P>|t|'] = p_table['P>|t|'].astype(float)
x_cols = list(p_table[p_table['P>|t|'] < 0.05].index)
x_cols.remove('Intercept')
print(len(p_table), len(x_cols))
print(x_cols[:5])
p_table.head()


# #checking for linearity of features 
# sns.jointplot('price','bathrooms', data=cleaned, kind='reg'); 
# sns.jointplot('price','bedrooms', data=cleaned, kind='reg'); 
# sns.jointplot('price','condition', data=cleaned, kind='reg');  
# sns.jointplot('price','grade', data=cleaned, kind='reg'); 
# sns.jointplot('price','sqft_living', data=cleaned, kind='reg'); 
# sns.jointplot('price','floors', data=cleaned, kind='reg'); 
# sns.jointplot('price','sqft_above', data=cleaned, kind='reg');  
# sns.jointplot('price','sqft_living15', data=cleaned, kind='reg'); 
# #checking for multicollinearity  
# feats = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15']
# corr = df[feats].corr()
# corr 
# 
# #considering r > 0.65 as multicolinear bathrooms & grade appears to be the only multicolinear item?   
# sns.heatmap(corr, center=0, annot=True) 
# Initial Model
# I would like to look at what renovatable (changable) feature most affects housing price. I am choosing to look only at 
# features that could be altered i.e. not view, zipcode, waterfront, yr_built, zipcode, lat, long, sqft_lot15 since those
# features cannot be changed. I am modeling my data this way because my stake holder in this project is a nonprofit/ city
# partnership looking to remodel houses at affordable prices & build equity in communities. 
# 
# #The problem
# outcome = 'price'
# x_cols = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15']  
# 
# #create a seprate df before changing values 
# inital_df = df.copy()
# 
# for col in x_cols: 
#     inital_df[col] = (inital_df[col] - df[col].mean())/inital_df[col].std()
# inital_df.head() 
# 
# #fitting the model 
# predictors = '+'.join(x_cols)
# formula = outcome + '~' + predictors 
# model = ols(formula=formula, data=inital_df).fit()
# model.summary() 
# 
# #variance inflation factor  
# X = inital_df[x_cols] 
# vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
# list(zip(x_cols, vif)) 
# 
# #update model 
# outcome = 'price'
# x_cols =['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15'] 
# predictors = '+'.join(x_cols)
# formula = outcome + '~' + predictors 
# model = ols(formula=formula, data=inital_df).fit()
# model.summary() 
# 
# #pval of 0.0 means that we are certain this relationship is not due to chance.  
# Normalicy Check 
# fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True) 
# Homoscedacity Check 
# plt.scatter(model.predict(inital_df[x_cols]), model.resid)
# plt.plot(model.predict(inital_df[x_cols]), [0 for i in range(len(inital_df))]) 
# Model Refinement
# #data cut off point
# for i in range(90, 99): 
#     q = i / 100 
#     print('{} percentile: {}'.format(q, inital_df['price'].quantile(q=q))) 
# subset = inital_df[inital_df['price'] < 1260000]
# print('Percent removed:', (len(inital_df) - len(subset)) / len(df))
# outcome= 'price'
# x_cols = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15'] 
# predictors = '+'.join(x_cols)
# formula = outcome + '~' + predictors
# model = ols(formula=formula, data=subset).fit()
# model.summary() 
# 
# #recheck normality 
# fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True) 
# 
# #recheck homostatisity 
# plt.scatter(model.predict(subset[x_cols]), model.resid)
# plt.plot(model.predict(subset[x_cols]), [0 for i in range(len(subset))])
