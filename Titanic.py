#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('test.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.describe().T.plot()


# In[8]:


df.isnull().sum()


# In[9]:


((df.isnull().sum())/(len(df)))*100


# ### Handling Missing Values

# #### For 'Age' column

# In[10]:


df['Age']


# In[11]:


df['Age'].describe().T.plot()


# In[12]:


print(df['Age'].nunique())


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Age'].dropna(), kde=True)
plt.show()


# In[14]:


# Boxplot of Age by Pclass and Sex
sns.boxplot(data=df, x='Pclass', y='Age', hue='Sex')
plt.title('Age Distribution by Pclass and Sex')
plt.show()


# In[15]:


# Fill missing Age values based on Pclass and Sex
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[16]:


df.isnull().sum()


# #### For Fare column

# In[17]:


sns.histplot(df['Fare'].dropna(), kde=True)
plt.show()


# In[18]:


median_fare = df['Fare'].median()

# Fill the missing Fare value with the median
df['Fare'].fillna(median_fare, inplace=True)

# Display the updated DataFrame
print(df)


# In[19]:


df.isnull().sum()


# #### For Cabin column

# In[20]:


df.drop(columns=['Cabin'],inplace=True)


# In[21]:


df.isnull().sum()


# ### Outliers

# #### Detecting using IQR

# In[22]:




# Function to count outliers using the IQR method
def count_outliers_iqr(data):
    outlier_counts = {}
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)  # First quartile
        Q3 = data[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range
        
        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
        upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
        
        # Count outliers
        outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outlier_counts[col] = outlier_count  # Store the count of outliers for the column
    
    return outlier_counts

# Count outliers in the DataFrame
outlier_counts = count_outliers_iqr(df)

# Display the number of outliers for each column
for col, count in outlier_counts.items():
    print(f"Number of outliers in '{col}': {count}")


# In[23]:


Q1 = df['Age'].quantile(0.25)  # First quartile
Q3 = df['Age'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers in the Age column
age_outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

# Display the outliers
print("Outliers in the 'Age' column:")
print(age_outliers['Age'])


# In[24]:


# Calculate group medians
grouped_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()

# Define function to replace outliers with the group median
def replace_outliers_with_group_median(row):
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if row['Age'] < lower_bound or row['Age'] > upper_bound:
        return grouped_medians.loc[row['Pclass'], row['Sex']]
    else:
        return row['Age']

# Apply function to replace outliers in Age
df['Age'] = df.apply(replace_outliers_with_group_median, axis=1)


# In[25]:


# Calculate bounds for Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap Fare outliers
df['Fare'] = df['Fare'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))


# In[26]:


# Loop through SibSp and Parch
for col in ['SibSp', 'Parch']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))


# ### Feature Encoding

# In[27]:


from sklearn.preprocessing import LabelEncoder

# Example: Encoding 'Sex'
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male -> 1, Female -> 0


# In[28]:


# One-Hot Encoding 'Embarked' and 'Pclass'
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)


# In[ ]:




