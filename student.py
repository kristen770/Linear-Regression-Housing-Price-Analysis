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
# https://www.unitedstateszipcodes.org/98039/ 
# https://www.usmapguide.com/washington/seattle-zip-code-map  
# https://github.com/EricaSG/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320 
# https://github.com/saifzkb/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320 
# https://github.com/jari-el13/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320/blob/master/student.ipynb 
# https://github.com/kailakay/dsc-mod-2-project-v2-1-onl01-dtsc-ft-041320/blob/master/Q1.ipynb
# https://github.com/mclaurenhr/Mod02-Linear-Regression

# In[1]:


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


# ## Workspace Set Up

# In[109]:


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

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#pip install pandasql


# In[43]:


#Read in housing data & initialize it to a variable
raw_data_df = pd.read_csv("data/kc_house_data.csv") 
raw_data_df


# ## Data Cleaning 

# In[44]:


#explore inital dataset 
raw_data_df.info()   
# There are 21,597 Observations (each row is a house sale)
# Waterfront, view, year renovated have null values   
# Date is object  
# There are "?" in the 

len(raw_data_df[raw_data_df.duplicated()]) 
#0 all unique home sales 


# In[45]:


#looking at the nan values 
raw_data_df.isna().sum() 
#watefront 2,376/ 21,597 NAN 
#view 63/ 21,597 NAN 
#yr_renovated 3,842/21,597 NAN  


# In[46]:


#replace NAN values with 0 
df = raw_data_df.fillna(0) 
df 

#veriying NAN values are gone
df.isna().sum()


# In[47]:


#change the date row to a datetime object 
df['date'] = pd.to_datetime(df['date'])


# In[48]:


#looking at values in sqft_basement 
df['sqft_basement'].unique()
df = df.replace({'sqft_basement': {"?": 0.0}}) 

#set as float 
df['sqft_basement'] = df['sqft_basement'].astype('float64')


# ##### Data Decision Explaination: 
# There are too many rows with NAN (particularly in the waterfront column) to drop without affecting the dataset, given that if all three of these coloumns are indicators that not every house would have, and opperating under the assumption that if the house did have any one of there three 'features' homeowners would be motivated to list them - I am going to replace all NaN with a 0 to indicate that the house does not have that feature.

# ## General Exploration (Initail EDA)

# In[49]:


df.head()


# ##### Data Analysis 
# There appears to be a stronger correlation between bathrooms and price than with bedrooms. While there doesn't appear to be any linearity between condition and price there is a relatively strong one between grade and price. This is interesting when considering the stake holders I identified for my project (city / non profit liason to refurbush houses)

# In[ ]:


#what is the mean price of all houses? 
df['price'].mean() 
#$540,296.57

#what is the median price of all houses? 
df['price'].median()  
#$ 450,000.00


# # Feature Engineering 

# In[ ]:


#unique zipcodes 
df['zipcode'].unique()


# In[ ]:


#breaking up zipcode by seattle city limits  
#outside city line
non_city_options = [98028, 98074, 98053, 98003, 98198, 98038, 98007, 98019, 98002, 98040, 98092, 98030, 98052, 98027, 98058, 
98001, 98056, 98166, 98023, 98070, 98148, 98042, 98008, 98059, 98004, 98005, 98034, 98075, 98010, 98032, 98045, 98077,  
98065, 98029, 98006, 98022, 98033, 98024, 98011, 98031, 98072, 98188, 98014, 98055, 98039]
burbs = df[df['zipcode'].isin(non_city_options)]   
burbs


# In[ ]:


#inside city line
options = [98155, 98177, 98133, 98125, 98117, 98103, 98115, 98105, 98102, 98112, 98109, 98107, 98119, 
           98199, 98122, 98144, 98134, 98108, 98118, 98168, 98106, 98126, 98136, 98116, 98146, 98178, 98121,
           98101, 98154, 98104] 
seattle_proper = df[df['zipcode'].isin(options)] 
seattle_proper


# ##### Data Decision Explaination: 
# I decided to gather information for each zipcode's population density in 2015 so that I could build models based on this information later. It would make sense that a 3 bedroom house in a high density (urban/downtown) area would have a different value that a 3 bedroom house in a low density (rural/ farmland) are and thus if I want a more accurate model I cannot create them as the same. 

# In[ ]:


#creating a dataframe of each zipcode & population density data from 2015
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
pop_density_by_zip


# In[ ]:


#map population density onto the main dataframe 
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
df.head()


# In[ ]:


#subdivide my dataframe by density into roughly equal sections 
low_density = df.loc[df['pop_density'] < 2500] 
med_density = df.loc[(df['pop_density'] > 2500) & (df['pop_density'] < 5000)] 
high_density = df.loc[df['pop_density'] > 5000]


#  # EDA 

# ## Question 1: What are the price & housing grade distributions throughout King County? 

# ### Housing Grade by Zipcode

# In[126]:


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


# In[127]:


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


# In[128]:


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


# In[125]:


#Houses that are below the condition grade and housing grade
low_grade_condition = df.loc[(df['condition'] < 3) & (df['grade'] <= 5)] 
low_grade_condition.groupby('zipcode').count() 
#There are only 1 or 2 houses that fall below the condition and grade minimum in the city in 16 of the cities 72 zipcode 
# and 1 zipcode with 7 houses 98168


# ##### Data Decision Explanation: 
# There is a relatively high distribution of houses in the 98168 zipcode that are in both poor condition and recieved a low housing grade. I chose to look more closely at the overall make up of the houses listed in this zipcode.

# In[129]:


low_grade_condition.loc[low_grade_condition['zipcode'] == 98168]['price'].mean() 
#The median price of these homes is 117,071.43 


# In[130]:


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


# In[133]:


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

# In[ ]:


#price_by_zips = df.loc 
df.groupby('zipcode').mean()


# In[135]:


#swarmplot of grade by zipcode 
ax = sns.boxplot(x="zipcode", y='grade', data=zips_grade)
ax = sns.swarmplot(x='zipcode', y='grade', data=zips_grade, color = ".25")


# ## Question 2: What zipcodes arebeing renovated at the highest rates? 

# ## Question 3: What house fetures are most closely correlated to code compliance(higher housing grades)?

# # Linear Regression Model

# ### EDA 

# #### Look for Outliers

# In[122]:


#visualize the df with a scatter plot matrix 
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(20,18)) #alpha = amount of transparency applied 


# In[143]:


df_full.columns


# In[147]:


#look at z scores (relationship to the std & mean) 
#df_full = df_full.drop(['date', 'id'], axis=1)
z = np.abs(stats.zscore(df_full)) 
threshold = 3 #3 std away from them mean
print(np.where(z > 3))


# In[150]:


df_full


# In[148]:


cleaned = df_full[(z<3).all(axis=1)]
cleaned 
#This dropped 2,755 rows from the df (12.75% of the data)


# ###### Data Decision Analysis 
# Dropping the outling data that was higher than 3 standard deviations from the mean did loose alomst 13% of the data. This is not an insignifigant amount this data contained a large number of particularly expensive houses which was giving the data a heavy right tail. By dropping these we are model will become less acurate in the highest and lowest ends of the market.

# In[176]:


#seperating the value of the bedroom/bathroom   
cleaned['bathrooms'] = cleaned['bathrooms'].astype(str)

cleaned['bathrooms'] = cleaned['bathrooms'].str[1:4] 


# In[178]:


cleaned['bathrooms'] = cleaned['bathrooms'].astype(float)


# In[179]:


cleaned.head()


# ##### Data Decision Analysis 
# Assuming that the definiton of a full or "1.0" bathroom is that is has a sink, toliet, and shower (3 appliances) I am assuming that each of these whole bathrooms is correlated to a bedroom. In order to reduce the multicolinearity of these two columns I am going to leave only the .50 (just toliet and sink) and .25 (just sink) values in the column to indicate a value of bathrooms that are in addition to the bedroom bathroom.

# In[180]:


#histogram of each column  
df.hist(figsize = (20,18))


# ### Model Just Renovatable Features

# In[152]:


#checking for linearity of features 
sns.jointplot('price','bathrooms', data=cleaned, kind='reg'); 
sns.jointplot('price','bedrooms', data=cleaned, kind='reg'); 
sns.jointplot('price','condition', data=cleaned, kind='reg');  
sns.jointplot('price','grade', data=cleaned, kind='reg'); 
sns.jointplot('price','sqft_living', data=cleaned, kind='reg'); 
sns.jointplot('price','floors', data=cleaned, kind='reg'); 
sns.jointplot('price','sqft_above', data=cleaned, kind='reg');  
sns.jointplot('price','sqft_living15', data=cleaned, kind='reg');


# In[67]:


#checking for multicollinearity  
feats = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15']
corr = df[feats].corr()
corr 

#considering r > 0.65 as multicolinear bathrooms & grade appears to be the only multicolinear item?  


# In[68]:


sns.heatmap(corr, center=0, annot=True)


# #### Initial Model  
# I would like to look at what renovatable (changable) feature most affects housing price. I am choosing to look only at features that could be altered i.e. not view, zipcode, waterfront, yr_built, zipcode, lat, long, sqft_lot15 since those features cannot be changed. I am modeling my data this way because my stake holder in this project is a nonprofit/ city partnership looking to remodel houses at affordable prices & build equity in communities. 

# In[75]:


#The problem
outcome = 'price'
x_cols = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15']


# In[73]:


#create a seprate df before changing values 
inital_df = df.copy()


# In[78]:


for col in x_cols: 
    inital_df[col] = (inital_df[col] - df[col].mean())/inital_df[col].std()
inital_df.head()


# In[80]:


#fitting the model 
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors 
model = ols(formula=formula, data=inital_df).fit()
model.summary()


# In[84]:


#variance inflation factor  
X = inital_df[x_cols] 
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
list(zip(x_cols, vif))


# In[95]:


#update model 
outcome = 'price'
x_cols =['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15'] 
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors 
model = ols(formula=formula, data=inital_df).fit()
model.summary()


# In[ ]:


#pval of 0.0 means that we are certain this relationship is not due to chance. 


# #### Normalicy Check

# In[88]:


fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)


# #### Homoscedacity Check

# In[90]:


plt.scatter(model.predict(inital_df[x_cols]), model.resid)
plt.plot(model.predict(inital_df[x_cols]), [0 for i in range(len(inital_df))])


# #### Model Refinement 

# In[96]:


#data cut off point
for i in range(90, 99): 
    q = i / 100 
    print('{} percentile: {}'.format(q, inital_df['price'].quantile(q=q)))


# In[100]:


subset = inital_df[inital_df['price'] < 1260000]
print('Percent removed:', (len(inital_df) - len(subset)) / len(df))
outcome= 'price'
x_cols = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15'] 
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=subset).fit()
model.summary()


# In[101]:


#recheck normality 
fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)


# In[103]:


#recheck homostatisity 
plt.scatter(model.predict(subset[x_cols]), model.resid)
plt.plot(model.predict(subset[x_cols]), [0 for i in range(len(subset))])


# ## Full Linear Regression 

# In[198]:


#make a copy of the df
full_cleaned = cleaned.copy()


# In[199]:


full_cleaned.columns


# In[200]:


#define the problem 
outcome = 'price'
x_col = ['date', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']


# In[201]:


#train & test groups
train, test = train_test_split(df_full) 
print("Train:", len(train), "Test:", len(test))


# In[202]:


train.head()


# In[203]:


test.head()


# In[204]:


#checking for multicollinearity  
feats = ['bathrooms', 'bedrooms', 'condition', 'grade', 'sqft_living', 'floors',  'sqft_above', 'sqft_living15']
corr = df[feats].corr()
corr 


# In[205]:


sns.heatmap(corr, center=0, annot=True)


# In[206]:


#fit model 
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors 
model = ols(formula=formula, data=train).fit()
model.summary()


# In[195]:


outcome = 'price'
x_cols= ['date', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'] 


# In[196]:


X = cleaned[x_cols] 
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
list(zip(x_cols, vif))


# In[190]:


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


# In[ ]:




