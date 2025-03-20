#!/usr/bin/env python
# coding: utf-8


# Import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime


# In this project, we focused to answer the following questions:
# 1.Which store has minimum and maximum sales?
# 2.Which store has maximum standard deviation i.e., the sales vary a lot. Also, 3.find out the coefficient of mean to standard deviation
# 4.Which store/s has good quarterly growth rate in Q3’2012
# 5.Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together
# 6.Provide a monthly and semester view of sales in units and give insights
# Build prediction to forecast demand.

# *Data Understanding
# There are sales data available for 45 stores of Walmart in Kaggle. This is the data that covers sales from 2010-02-05 to 2012-11-01.
# 
# The data contains these features:
# 
# Store - the store number
# Date - the week of sales
# Weekly_Sales - sales for the given store
# Holiday_Flag - whether the week is a special holiday week 1 – Holiday week 0 – Non-holiday week
# Temperature - Temperature on the day of sale
# Fuel_Price - Cost of fuel in the region
# CPI – Prevailing consumer price index
# Unemployment - Prevailing unemployment rate


# Load dataset
data = pd.read_csv(r"C:\Users\91832\Downloads\Walmart_Store_sales.csv")
data
print(data.head())  # To check if the file loads properly


# DATA PREPARATION
# Convert date to datetime format and show dataset information
data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")
data.info()

# checking for missing values
data.isnull().sum()



# Splitting Date and create new columns (Day, Month, and Year)
data["Day"]= pd.DatetimeIndex(data['Date']).day
data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Year'] = pd.DatetimeIndex(data['Date']).year
data

plt.figure(figsize=(15,7))

# Sum Weekly_Sales for each store, then sortded by total sales
total_sales_for_each_store = data.groupby('Store')['Weekly_Sales'].sum().sort_values() 
total_sales_for_each_store_array = np.array(total_sales_for_each_store) # convert to array

# Assigning a specific color for the stores have the lowest and highest sales
clrs = ['lightsteelblue' if ((x < max(total_sales_for_each_store_array)) and (x > min(total_sales_for_each_store_array))) else 'midnightblue' for x in total_sales_for_each_store_array]


ax = total_sales_for_each_store.plot(kind='bar',color=clrs);

# store have minimum sales
p = ax.patches[0]
print(type(p.get_height()))
ax.annotate("The store has minimum sales is 33 with {0:.2f} $".format((p.get_height())), xy=(p.get_x(), p.get_height()), xycoords='data',
            xytext=(0.17, 0.32), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='center', verticalalignment='center')


# store have maximum sales 
p = ax.patches[44]
ax.annotate("The store has maximum sales is 20 with {0:.2f} $".format((p.get_height())), xy=(p.get_x(), p.get_height()), xycoords='data',
            xytext=(0.82, 0.98), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='center', verticalalignment='center')


# plot properties
plt.xticks(rotation=0)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.title('Total sales for each store')
plt.xlabel('Store')
plt.ylabel('Total Sales');


# Q1: Which store has minimum and maximum sales?

# Q2: Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation?

# Which store has maximum standard deviation
data_std = pd.DataFrame(data.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
print("The store has maximum standard deviation is "+str(data_std.head(1).index[0])+" with {0:.0f} $".format(data_std.head(1).Weekly_Sales[data_std.head(1).index[0]]))


# Distribution of store has maximum standard deviation
plt.figure(figsize=(15,7))
sns.distplot(data[data['Store'] == data_std.head(1).index[0]]['Weekly_Sales'])
plt.title('The Sales Distribution of Store #'+ str(data_std.head(1).index[0]));


# Coefficient of mean to standard deviation
coef_mean_std = pd.DataFrame(data.groupby('Store')['Weekly_Sales'].std() / data.groupby('Store')['Weekly_Sales'].mean())
coef_mean_std = coef_mean_std.rename(columns={'Weekly_Sales':'Coefficient of mean to standard deviation'})
coef_mean_std

# Distribution of store has maximum coefficient of mean to standard deviation
coef_mean_std_max = coef_mean_std.sort_values(by='Coefficient of mean to standard deviation')
plt.figure(figsize=(15,7))
sns.distplot(data[data['Store'] == coef_mean_std_max.tail(1).index[0]]['Weekly_Sales'])
plt.title('The Sales Distribution of Store #'+str(coef_mean_std_max.tail(1).index[0]));


# Q3: Which store/s has good quarterly growth rate in Q3’2012

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))

# Sales for third quarter in 2012
Q3 = data[(data['Date'] > '2012-07-01') & (data['Date'] < '2012-09-30')].groupby('Store')['Weekly_Sales'].sum()

# Sales for second quarter in 2012
Q2 = data[(data['Date'] > '2012-04-01') & (data['Date'] < '2012-06-30')].groupby('Store')['Weekly_Sales'].sum()

# Plotting both Q2 and Q3 sales on the same figure
ax = Q3.plot(kind='bar', alpha=0.7, label="Q3' 2012")  # First plot
Q2.plot(kind='bar', alpha=0.4, ax=ax, label="Q2' 2012")  # Second plot on the same axis

plt.legend()
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.title("Comparison of Q2 and Q3 Sales in 2012")
plt.show()

#  store/s has good quarterly growth rate in Q3’2012 - .sort_values(by='Weekly_Sales')
print('Store have good quarterly growth rate in Q3’2012 is Store '+str(Q3.idxmax())+' With '+str(Q3.max())+' $')


# Q4: Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together
# Holiday Events:
# 
# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
# 
# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
# 
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
# 
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')

# Group sales by date
total_sales = data.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Holiday dates
Super_Bowl = ['12-2-2010', '11-2-2011', '10-2-2012']
Labour_Day = ['10-9-2010', '9-9-2011', '7-9-2012']
Thanksgiving = ['26-11-2010', '25-11-2011', '23-11-2012']
Christmas = ['31-12-2010', '30-12-2011', '28-12-2012']

# Convert holiday dates to datetime
def plot_line(df, holiday_dates, holiday_label):
    fig, ax = plt.subplots(figsize=(15,5))  
    ax.plot(df['Date'], df['Weekly_Sales'], label=holiday_label)

    # Convert holiday dates
    holiday_dates = pd.to_datetime(holiday_dates, format='%d-%m-%Y', errors='coerce')

    # Add vertical lines for holidays
    for day in holiday_dates:
        plt.axvline(x=day, linestyle='--', c='r')

    # Format x-axis
    plt.title(holiday_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))  # Adjust as needed
    plt.gcf().autofmt_xdate(rotation=90)
    plt.legend()
    plt.show()

# Plot each holiday
plot_line(total_sales, Super_Bowl, 'Super Bowl')
plot_line(total_sales, Labour_Day, 'Labour Day')
plot_line(total_sales, Thanksgiving, 'Thanksgiving')
plot_line(total_sales, Christmas, 'Christmas')


# The sales increased during thanksgiving. And the sales decreased during christmas.
import pandas as pd

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')

# Convert Super_Bowl dates
Super_Bowl = pd.to_datetime(['12-2-2010', '11-2-2011', '10-2-2012'], format='%d-%m-%Y', errors='coerce')

# Filter data
filtered_data = data.loc[data['Date'].isin(Super_Bowl)]
print(filtered_data)

# Yearly Sales in holidays
Super_Bowl_df = pd.DataFrame(data.loc[data.Date.isin(Super_Bowl)].groupby('Year')['Weekly_Sales'].sum())
Thanksgiving_df = pd.DataFrame(data.loc[data.Date.isin(Thanksgiving)].groupby('Year')['Weekly_Sales'].sum())
Labour_Day_df = pd.DataFrame(data.loc[data.Date.isin(Labour_Day)].groupby('Year')['Weekly_Sales'].sum())
Christmas_df = pd.DataFrame(data.loc[data.Date.isin(Christmas)].groupby('Year')['Weekly_Sales'].sum())

Super_Bowl_df.plot(kind='bar',legend=False,title='Yearly Sales in Super Bowl holiday') 
Thanksgiving_df.plot(kind='bar',legend=False,title='Yearly Sales in Thanksgiving holiday') 
Labour_Day_df.plot(kind='bar',legend=False,title='Yearly Sales in Labour_Day holiday')
Christmas_df.plot(kind='bar',legend=False,title='Yearly Sales in Christmas holiday')


# Q5: Provide a monthly and semester view of sales in units and give insights

# Monthly view of sales for each years
plt.scatter(data[data.Year==2010]["Month"],data[data.Year==2010]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2010")
plt.show()

plt.scatter(data[data.Year==2011]["Month"],data[data.Year==2011]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2011")
plt.show()

plt.scatter(data[data.Year==2012]["Month"],data[data.Year==2012]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2012")
plt.show()


# Monthly view of sales for all years
# Ensure Month is an integer
data["Month"] = data["Month"].astype(int)

# Aggregate sales by month
monthly_sales = data.groupby("Month")["Weekly_Sales"].sum().reset_index()

# Plot the bar chart
plt.figure(figsize=(10,6))
plt.bar(monthly_sales["Month"], monthly_sales["Weekly_Sales"])

# Labels and title
plt.xlabel("Months", fontsize=14)
plt.ylabel("Total Weekly Sales", fontsize=14)
plt.title("Monthly View of Sales", fontsize=16)

# Formatting the x-axis
plt.xticks(range(1, 13), fontsize=12)
plt.yticks(fontsize=12)

# Remove top and right borders for a clean look
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.show()


# Yearly view of sales
plt.figure(figsize=(10,6))
data.groupby("Year")[["Weekly_Sales"]].sum().plot(kind='bar',legend=False)
plt.xlabel("years")
plt.ylabel("Weekly Sales")
plt.title("Yearly view of sales");


# Build prediction models to forecast demand (Modeling)

# find outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(data[column], ax=axs[i])


# drop the outliers     
data_new = data[(data['Unemployment']<10) & (data['Unemployment']>4.5) & (data['Temperature']>10)]
data_new


# check outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = data_new[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(data_new[column], ax=axs[i])


# BUILD MODEL


# Import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression



# Select features and target 
X = data_new[['Store','Fuel_Price','CPI','Unemployment','Day','Month','Year']]
y = data_new['Weekly_Sales']

# Split data to train and test (0.80:0.20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# Linear Regression model

import seaborn as sns

# Linear Regression model
print('Linear Regression:')
print()

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('Accuracy:', reg.score(X_train, y_train) * 100)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Scatter plot (fixed)
sns.scatterplot(x=y_pred, y=y_test)

import seaborn as sns

# Random Forest Regressor Model
print('Random Forest Regressor:')
print()

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('Accuracy:', rf.score(X_train, y_train) * 100)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Scatter plot (Fixed)
sns.scatterplot(x=y_pred, y=y_test)












