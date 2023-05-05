#!/usr/bin/env python
# coding: utf-8

# # Analyzing Categorical Data 
# 
# In this project, we'll be working with categorical data and will be using a subset of data from the following data set: (https://www.kaggle.com/datasets/norc/general-social-survey?select=gss.csv).
# 
# After cleaning the data, we will use some visualizations tools. We also had used statsmodels for a special type of categorical plot.

# In[1]:


# Import packages
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Read in csv as a DataFrame and preview it
df = pd.read_csv("/Users/antoniogondim/Downloads/gss_sub.csv")
df


# In[46]:


df.info()


# In[2]:


df=df.drop('inequality', axis=1)
#Too many null values

    


# 

# In[3]:


df=df.dropna()
df.shape


# Above we see that our DataFrame contains `float64` column (numerical data), as well as a number of `object` columns, i.e object data types contain strings.

# df.describe() method with the `include` parameter to select a particular DataType (in this case `"O"`). This returns the count, number of unique values, the mode, and frequency of the mode for each column having object as data type.

# In[49]:


df.describe(include="O")


# In[50]:


df["environment"].value_counts()


# ## Manipulating categorical data
# 
# - The categorical variable type can be useful, especially here:
#     - It is possible to specify a precise order to the categories when the default order may be incorrect (e.g., via alphabetical).
#     - Can be compatible with other Python libraries.
# 
# Let's take our existing categorical variables and convert them from strings to categories. Here, we use [`.select_dtypes()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html) to return only object columns, and with a dictionary set their type to be a category.

# In[4]:


# Create a dictionary of column and data type mappings
conversion_dict = {k: "category" for k in df.select_dtypes(include="object").columns}

# Convert our DataFrame and check the data types
df = df.astype(conversion_dict)
df.info()


# Already we can see that the memory usage of the DataFrame has been halved from 7 mb to about 4 mb, optmizing the data.

# ### Cleaning up the `labor_status` column
# To analyze the relationship between employment and attitudes over time, we need to clean up the `labor_status` column. We can preview the existing categories using `.categories`.

# In[5]:


df["labor_status"].values


# In[6]:


df["labor_status"].value_counts()


# Let's collapse some of these categories. The easiest way to do this is to replace the values inside the column using a dictionary, and then reset the data type back to a category.

# In[7]:


# Create a dictionary of categories to collapse
new_labor_status = {"UNEMPL, LAID OFF": "UNEMPLOYED", 
                    "TEMP NOT WORKING": "UNEMPLOYED",
                    "WORKING FULLTIME": "EMPLOYED",
                    "WORKING PARTTIME": "EMPLOYED"
                   }

# Replace the values in the column and reset as a category
df["labor_status_clean"] = df["labor_status"].replace(new_labor_status).astype("category")
print(df.dtypes)
# Preview the new column
df["labor_status_clean"].value_counts()


# ### Reordering categories
# Another potential issue is the order of our opinion variables (`environment`, `law_enforcement`, and `drugs`). These are ordinal variables, or categorical variables with a clear ordering or ranking. However, these orders are not currently set. 
# 
# This will affect use later when we go to visualize our data. We can also take the opportunity to drop some unwanted categories.

# In[55]:


df["environment"].values


# Let's loop through the three variables and give them all an order. While we're at it, let's drop two categories that don't have any use for us: "DK" (don't know) and "IAP" (inapplicable). By removing them as categories, we set them to null so they won't be counted in the final analysis.

# In[9]:


# Set the new order
new_order = ["TOO LITTLE", "ABOUT RIGHT", "TOO MUCH", "DK", "IAP"]
categories_to_remove = ["DK", "IAP"]

# Loop through each column
for col in ["environment", "law_enforcement", "drugs"]:
    # Reorder and remove the categories
    df[col + "_clean"] = df[col].cat.reorder_categories(new_order, ordered=True)
    df[col + "_clean"] = df[col + "_clean"].cat.remove_categories(categories_to_remove)

# Preview one of the columns' categories
df["environment_clean"].cat.categories


# Now let's also apply these steps to education level in one go: collapsing, removing, and reording.

# In[10]:


df['degree'].values #let's reorder that and remove 'DK'


# In[11]:


# Define a dictionary to map old degree categories to new ones
new_degree = {"LT HIGH SCHOOL": "HIGH SCHOOL", 
              "BACHELOR": "COLLEGE/UNIVERSITY",
              "GRADUATE": "COLLEGE/UNIVERSITY",
              "JUNIOR COLLEGE": "COLLEGE/UNIVERSITY"}

# Replace old degree categories with new ones and convert to categorical data type
df["degree_clean"] = df["degree"].replace(new_degree).astype("category")

# Remove "DK" category from degree_clean column
df["degree_clean"] = df["degree_clean"].cat.remove_categories(["DK"])

# Reorder degree_clean categories and set as ordered
df["degree_clean"] = df["degree_clean"].cat.reorder_categories(["HIGH SCHOOL", "COLLEGE/UNIVERSITY"], ordered=True)

# Preview the new column
df["degree_clean"].value_counts()


# 
# By [`IntervalIndex`](https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.html) we set cutoff ranges for the `year`. We then use [`pd.cut()`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html) to cut our `year` column by these ranges, and set labels for each range.

# In[13]:


decade_boundaries = [(1970, 1979),(1979,1989) , (1989, 1999), (1999, 2009), (2009, 2019)]
# Set the bins and cut the DataFrame

bins = pd.IntervalIndex.from_tuples(decade_boundaries)

decade_labels = {bins[0]:'1970s',
                 bins[1]: '1980s',
                 bins[2]:'1990s',
                 bins[3]:'2000s',
                bins[4]:'2010s'}


print(bins)
df["decade"] = pd.cut(df["year"], bins)
print(df['decade'].values, df['decade'].dtypes)

# Rename each of the intervals of decade_boundaries as the decades on decade_labels
df['decade']=df["decade"].replace(decade_labels)#.astype('category')#PQ NAO TA FUNCIONANDO??????

# Preview the new column
df[['year','decade']]


# ## Visualization

# In[14]:


# Create a new figure object
fig = px.bar(df["labor_status"].value_counts(),
             template="plotly_white"
            )

# Hide the legend and show the plot
fig.update_layout(showlegend=False)
fig.show()


# Let's change the orientation of the plot and add a title, for a better perspective.

# In[15]:


# Create a new figure object
fig = px.bar(df["labor_status"].value_counts(ascending=True),
             template="plotly_white",
             orientation="h",
             title="Labor status by count"
            )

# Hide the legend and show the plot
fig.update_layout(showlegend=False)
fig.show()


# ### Bar charts 

# In[33]:


## Aggregate household size by year
household_by_decade = df.groupby("decade",as_index=False)["household_size"].mean()
household_by_decade


# In[18]:


# Create a new figure object
fig = px.bar(household_by_decade,
             x="decade",
             y="household_size",
             template="plotly_white",
             title="Average household size by decade"
            )

fig.show()


# ### Boxplots
# 

# In[19]:


# Create a new figure object
fig = px.box(df,
             x="age",
             y="labor_status_clean",
             template="plotly_white"
            )

fig.show()


# ### Mosaic plots
# visualize the relationship between two categorical variables. One way to do this is a frequency table, which will give the counts across the different combinations of the two variables.
# create a frequency table using [`pd.crosstab()`](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)

# In[22]:


pd.crosstab(df["degree_clean"], df["law_enforcement_clean"])


# In[23]:


# Create a mosaic plot and show it
mosaic(df, 
       ['degree_clean', 'law_enforcement_clean'], 
       title='Law enforcement opinions by degree')

plt.show()


# ### Line charts
# The final plot type we will cover is a line plot. Line plots often (but not always!) show the relationship between time and a numerical variable. Adding in a categorical variable can be a great way to enrich a line plot and provide other information.
# 
# Here, we use the `.value_counts()` method as an aggregation function, and use this in combination with a Plotly [`line_plot()`](https://plotly.com/python/line-charts/) to visualize the trend in marital statuses over the years.

# In[24]:


# Group the dataframe by year and marital status, and calculate the normalized value counts
marital_rates = df.groupby(["year"], as_index=False)["marital_status"].value_counts(normalize=True)

# Display the resulting DataFrame
marital_rates


# In[25]:


# Create a new figure object
fig = px.line(marital_rates,
              x="year",
              y="proportion",
              color="marital_status",
              template="plotly_white",
              title="Marital status over time"
             )

# Update the y-axis to show percentages
fig.update_yaxes(tickformat=".0%")

# Show the plot
fig.show()

