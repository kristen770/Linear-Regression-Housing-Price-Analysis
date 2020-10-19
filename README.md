# Phase 2 Final Project: King County Housing Data Anaylsis and Regression  
By: Kristen Davis 
<p align="center">
  <img width="850" height="250" src="/Photos/image0%20(1).jpeg">
</p>
 
 [](/Photos/image0%20(1).jpeg)

# Project Overview  
This project uses a modified version of King County House Sale data. The project comprised of two major parts: EDA & building a linear regression model. This data includes housing price, data sold, county house condition and grade rating, longitude & latitude points & number of bedrooms and bathrooms among other columns. I was specifically intersted in the homes being sold that fell on the low end of this spectrum (i.e. low quality, less expensive houses.)


# Business Problem  
In an effort to increase access to quality affordable housing, keep people rooted in their neighborhoods and fight the effect of over development King County and Habitat for Humanity are creating a public private initiative aimed at making data driven decisions to inform house acquisition and repairs in the greater Seattle area.  

# EDA  
Given the defined buisness problem I wanted to focus my EDA and define a narrow window exploration that would lead to strong data insights and analysis. 

## Question 1: What is the distribution of housing grade in the data? 
The majoritiy of houses in this data set are in the "average" grade range (7/8) there is a larger distribution of houses on the high end of the grading scale (9-13) than on the lower end (0-6). houses with a grade of 3 or 13 could be identified as outliers.    
<p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis.jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis.jpg)

## Question 2: What is the distribution of condition in the data? 
The majority of the houses in this data set are "good" condition (3/5) which is the average quality of a a home. There is a heavy tail of houses above that and 199 houses below that level. 

Looking at the distribution of these houses by zipcode, zipcode 98168 emerged as the zipcode with the highest percentage of these "poor condtion homes".   

<p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(1).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(1).jpg)

## Question 3: What is the break down of 98168? 
Notcing this trend in "poor condition" homes in on specific zipcode I wanted to break down the zipcode even further and look for any trends. It emerged that this zipcode has some very low quality inexpensive homes and some very expensive and "fancy" homes. This is particularly interesting for my stakeholders as it looks as though this zipcode is going through some major shifts in the make up of the community.  

<p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(2).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(2).jpg)

## Question 4: When were renovated houses renovated in 98168? 
This trend is even more highlighted when you look at when the houses that have been sold in this neighborhood were renovated. There has been a %400 increase in the number of houses renvoated since 2000, when you consider that these houses were then sold in 2015 that adds to the idea that these houses are potentially being "flipped" and houses similar to the ones represented in this sale data could be targeted by my stakeholders in order to preserve neighborhood identity.  
 <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(3).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(3).jpg) 
 
# Linear Regressions 
For my final model I initially filered my data to be houses that would be of primary interest to my stakeholders (cutting off houses above 1M for example). I think transformed some of the non normally distributed data and looked to build three distinctive models to compare.  
  <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(4).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(4).jpg)   
 
The final model I settled on used view, grade, lot square footage, and waterfront visibility to predict prices. The R2 was 0.386 and, while the warnings still indicated potential multicolinearity the VIF scores were all under 2: 

   <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(6).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(6).jpg) 
 
 The model was also fairly normal:  
  
   <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(7).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(7).jpg) 
 
 Finally for my stakeholders I used a graph to look at how sqftage and grade correlated to price and found that these were strongly predictive. 
 
   <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(5).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(5).jpg)     
 
 Finally I decided to remove the y-intercept to see if it would improve my model which it did signifigantly bumping my Rsquared up to 0.894. 
 
  <p align="center">
  <img width="550" height="250" src="/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(8).jpg">
</p>
 
 [](/Photos/Phase%202%20Final%20Project%20-%20Kristen%20Davis%20(8).jpg) 
 

# Conclusions  
Based on this analysis my stake holders, King County and Habitat for Humanity, should look at houses that are at or below the minimum grade in housing condition and/or grade in zipcodes such as 98168 to buy or select for renovation projects. When renovating or updating houses they should focus on improving the overall house's grade and increasing sq footage where possible to increase the house's price. 

# Future Work 
1. Explore new data - the King County Website has extensive housing data [here](https://info.kingcounty.gov/assessor/DataDownload/default.aspx) I would dig into and see how it could enhance the data I already have 
2. A/B Testing - Look at the effect of an additional bedroom or bathroom on price, preform similar A/B testing on varying grades & condition scores
3. Model Refinement - Continue to iterate over OLS model, consider new features to engineer and other ways of preparring the data. 
4. Geographic Considerations - Map demographic information with folium or use the lat/ lon data points in other meaningful ways. 


### [Blog Post](https://medium.com/@kristendavis27/starting-a-project-pick-an-angle-7a016c9b65b2) 
### [Project Recording](https://youtu.be/mY8beWKuIHs)
