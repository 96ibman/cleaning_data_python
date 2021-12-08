#!/usr/bin/env python
# coding: utf-8

# # 1. Data Type Constraints

# **Importing Data**
# <br>
# This dataset is about ride sharing, it contains information about each ride trip.<br>
# <br>information available are:<br>
# - rideID
# - duration
# - source station ID
# - souce station name
# - destination station ID
# - destination station name 
# - bike IDF
# - user type
# - user birth year
# - user gender

# In[1]:


import numpy as np
import pandas as pd
filepath = "D:/cleaning_data/data/ride_sharing_new.csv"
ride_sharing = pd.read_csv(filepath)
ride_sharing.head()


# **Ride Duration**
# <br>
# 
# The very first thing to spot as a data analyst, is the duration column values, as you can see the values contain "minutes", which is not what it should be, we need this to be pure `numerical` data type, not `string`!

# In[7]:


ride_sharing['duration']


# **Let's handle this!**
# <br>
# 
# **1. First:** remove the text minutes from every value
#    - we will use the function `strip` from the `str` module
#    - and store the new value in a new column: `duration_trim`

# In[8]:


ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')
ride_sharing['duration_trim']


# **2. Second:** convert the data type to integer
# - we will apply the `astype` method into `duration_trim`
# - and store the new values in a new column: `duration_time`

# In[9]:


# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')
ride_sharing['duration_time']


# **3. Check with an assert statement**

# In[10]:


assert ride_sharing['duration_time'].dtype == 'int'


# We can now get insight about the average duration time!

# In[12]:


print("Average Ride Duration:")
print(str(ride_sharing['duration_time'].mean()) + ' minutes')


# **User Type**
# <br>
# Let's have a look at the user type column by calling the `describe` method

# In[2]:


ride_sharing['user_type'].describe()


# When we called the describe method, it turned out that pandas treates this information as `float`, while its a `categorical` information.
# <br>
# Errors with regards to **data type constraints** are very common and important to handle in the data cleaning process.
# <br>
# 
# `user_type` shouldn't be treated as `float`, it is **categorical**
# <br>
# 
# The `user_type` column contains information on whether a user is taking<br>a free ride and takes on the following values:
# 
#     1 for free riders.
#     2 for pay per ride.
#     3 for monthly subscribers.

# **Let's fix this and convert the data type column to categorical**

# In[3]:


# Convert user_type to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')


# **Let's check with as assert statement**

# In[4]:


# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'


# **Let's double-check manually**

# In[6]:


# Print new summary statistics 
ride_sharing['user_type_cat'].describe()


# **Great!** 
# <br>
# Take a look at the new summary statistics, it seems that most users are pay per ride users because the top category is 2

# **Problems with data types are solved!**

# # 2. Inconsistent Categories

# **Importing Data**
# <br>
# This dataset is about airline flights, it contains people survey resposnses about a flight.<br>
# <br>information available are:<br>
# - response ID
# - flight ID
# - day
# - airline
# - destination country
# - destination region
# - boarding area
# - departure time
# - waiting minutes
# - how clean the plan was
# - how safe the flight was
# - satisfaction level of the flight

# In[13]:


filepath = "D:/cleaning_data/data/airlines_final.csv"
airlines = pd.read_csv(filepath)
airlines.head()


# **How to check for inconsistencies in categorical variables?**
# <br>
# Applying the `unique` method on the categorical feature to spot errors

# **Let's make a list of the categorical features**

# In[32]:


airlines.columns


# In[33]:


categorical_features = ['day', 'airline', 'destination', 'dest_region',
       'dest_size', 'boarding_area', 'cleanliness',
       'safety', 'satisfaction']


# **Build a function to check the unique values**

# In[47]:


def check_unique(col):
    print('------------------------------------------------------------------')
    print(f"Column: {col}")
    print(airlines[col].unique())


# **Looping over the list**

# In[48]:


for col in categorical_features:
    check_unique(col)
print('------------------------------------------------------------------')


# **Every thing looks fine, however, there is something wrong with `dest_region` and `dest_size`**
# <br>
# **Let's solve them**

# `dest_region` contains region 'Europe' and region 'eur' which are the same, it also contains 'EAST US' and 'East US' which are also the same but different values because of the upper/lower case.
# <br>
# **Let's lower them all**

# In[49]:


airlines['dest_region'] = airlines['dest_region'].str.lower() 
airlines['dest_region'].unique()


# **Let's replace 'eur' with 'europe'**

# In[50]:


airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})
airlines['dest_region'].unique()


# **the `dest_region` column solved!**
# <br>
# 
# **Let's see `dest_size`**

# In[51]:


airlines['dest_size'].unique()


# As you can tell, there is a **spacing** issue with this columns<br>
# How can we solve this?<br>
# ...<br>
# Exactly! with the `strip` method from the `str` module

# In[52]:


airlines['dest_size'] = airlines['dest_size'].str.strip()
airlines['dest_size'].unique()


# **Inconsistent categories issue has been solved!**

# # 3. Cross Field Validation

# Cross-Field Validation is the use of multiple fields in your dataset to sanity check the integrity of your data

# In[75]:


filepath = "D:/cleaning_data/data/banking_dirty.csv"
banking = pd.read_csv(filepath)
banking.head()


# **The dataset**
# <br>
# This dataset contains information about bank acounts investments.<br>
# The features are:<br>
# - ID
# - customer ID
# - customer birth date
# - customer age
# - account amount
# - investment amount
# - first fund amount
# - second fund amount
# - third fund amount
# - account open date
# - last transaction date

# **Where cross-field validation (CFV) can be applied?**
# <br>
# We can apply CFV to two columns:
# 1. Age 
# 2. inv_amount
# 
# **Age**
# <br>
# We can check the validity of the age by computing it manually from the birth date and check for errors
# <br> <br>
# **inv_amount**
# <br>
# We can apply CFV to the whole amount by manually sum all of the four funds and check if they sum up to the whole

# **1. CFV for Age**

# In[76]:


banking['birth_date']


# Alright, first, we have to convert this to datatime

# In[77]:


banking['birth_date'] = pd.to_datetime(banking['birth_date'])


# In[78]:


banking['birth_date']


# Then we will find ages manually

# In[89]:


import datetime as dt
today = dt.date.today()
ages_manual = (today.year - 1) - banking['birth_date'].dt.year
ages_manual


# Find consistent and inconsistent ages

# In[91]:


age_equ = ages_manual == banking['Age']
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]


# In[92]:


print("Number of inconsistent ages: ", inconsistent_ages.shape[0])


# In[93]:


banking[~age_equ]


# **2. CFV for inv_amount**

# Store the partial amounts columns in a list

# In[94]:


fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']


# Find consistent and inconsistent amounts

# In[95]:


# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis = 1) == banking['inv_amount']

# Store consistent and inconsistent data
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]

print("Number of inconsistent investments: ", inconsistent_inv.shape[0])


# In[96]:


banking[~inv_equ]


# ### What to do with inconsistencies?
# There is no *one size fits all* solution, the best solution requires an in-depth understanding of the dataset.<br>
# We can decide to either<br>
# - drop inconsistent data
# - deal with inconsistent data as missing and impute them
# - apply some rules due to domain knowledge

# # That's it!
# ## Share this with others!
# #### Ibrahim M. Nasser
