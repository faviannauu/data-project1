import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import math

# Helper Functions

# Data wrangling has yet to be done, so here's the data wrangling function
def clean_data(df):
    # Since the only thing needed to be cleaned is the outliers in specific attributes
    outliers = {"hum" : "float", "windspeed" : "float", "casual" : "integer"}

    # Loop through attributes
    for key, value in outliers.items():
        q25, q75 = np.percentile(df[key], 25), np.percentile(df[key], 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        minimum, maximum = q25 - cut_off, q75 + cut_off

        if value == "float":
            df.loc[df[key] < minimum, key] = minimum
            df.loc[df[key] > maximum, key] = maximum
        elif value == "integer":
            df.loc[df[key] < minimum, key] = math.ceil(minimum)
            df.loc[df[key] > maximum, key] = math.floor(maximum)
    return df

def weather_data(df):
    weather_df = df[["weathersit", "cnt"]].groupby("weathersit").mean()
    return weather_df

def temp_data(df):
    temp_df = df[["temp", "cnt"]]
    return temp_df

def hum_data(df):
    hum_df = df[["hum", "cnt"]]
    return hum_df

def windspeed_data(df):
    windspeed_df = df[["windspeed", "cnt"]]
    return windspeed_df

def workday_data(df):
    workday_df = df[["workingday", "casual", "cnt"]].groupby("workingday")
    return workday_df

def weekday_data(df):
    weekday_df = df[["weekday", "casual", "cnt"]].groupby("weekday")
    return weekday_df

# Start
# Load main data
main_df = pd.read_csv("main_data.csv")

# Cleaning the data
main_df = clean_data(main_df)

# Making sure the datetime format is correct
main_df["dteday"] = pd.to_datetime(main_df["dteday"])

# Filtering the data
min_date = main_df["dteday"].min()
max_date = main_df["dteday"].max()

with st.sidebar:
    # taking start_date and end_date as an input to filter the time span used in the main dataframe
    start_date, end_date = st.date_input(
        label='Time span',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

filtered_df = main_df[(main_df["dteday"] >= str(start_date)) &  (main_df["dteday"] <= str(end_date))]

weather_df = weather_data(filtered_df)
temp_df = temp_data(filtered_df)
hum_df = hum_data(filtered_df)
windspeed_df = windspeed_data(filtered_df)
workday_df = workday_data(filtered_df)
weekday_df = weekday_data(filtered_df)

# Plotting

st.header('Data analysis of Bike Sharing Dataset')

st.subheader('Rented Bike')
col1, col2 = st.columns(2)

with col1:
    total_rented = filtered_df.cnt.sum()
    st.metric("Total Bike Rented", value=total_rented)

with col2:
    total_casual = filtered_df.casual.sum()
    st.metric("Bike Rented by Casual Users", value=total_casual)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(filtered_df["dteday"], filtered_df["cnt"], marker='o',  linewidth=2, color="#90CAF9")
ax.plot(filtered_df["dteday"], filtered_df["casual"], marker='o',  linewidth=2, color="#5EBD77")
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

st.subheader('Weather condition')
st.text("Note: The value below are normalized values, the visualization provides a comparison of said condition")
col1, col2, col3 = st.columns(3)

with col1:
    avg_temp = filtered_df.temp.mean()
    st.metric("Average Temperature", value=avg_temp)

with col2:
    avg_hum = filtered_df.hum.mean()
    st.metric("Average Humidity", value=avg_hum)

with col3:
    avg_wind = filtered_df.windspeed.mean()
    st.metric("Average Windspeed", value=avg_wind)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(filtered_df["dteday"], filtered_df["temp"], marker='o',  linewidth=2, color="#90CAF9")
ax.plot(filtered_df["dteday"], filtered_df["hum"], marker='o',  linewidth=2, color="#5EBD77")
ax.plot(filtered_df["dteday"], filtered_df["windspeed"], marker='o',  linewidth=2, color="#FFC0CB")
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

st.subheader('Rented Bike on each Weather Condition')
st.text("Weather percentage on the filtered time span:")
st.text("dark to light: clear, misty, light rain or snow")

fig, ax = plt.subplots(figsize=(10, 6))
ax.pie(
    x = weather_df["cnt"],
    autopct = '%1.1f%%',
    colors = ["#5EBD77", "#90EE90", "#D0FFE0"],
    wedgeprops = {'width': 0.6}
)
st.pyplot(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 6))
sns.regplot(
    x = filtered_df["temp"],
    y = filtered_df["cnt"],
    scatter_kws={'s':6},
    ax=axs[0]
)
axs[0].set_ylabel("count")
axs[0].set_title("Rental count with the temperature Correlation")

sns.regplot(
    x = filtered_df["hum"],
    y = filtered_df["cnt"],
    scatter_kws={'s':6, 'color': 'green'},
    line_kws={"color": "green"},
    ax=axs[1]
)
axs[1].set_ylabel("count")
axs[1].set_title("Rental count with the hum Correlation")

sns.regplot(
    x = filtered_df["windspeed"],
    y = filtered_df["cnt"],
    scatter_kws={'s':6, 'color': 'orange'},
    line_kws={"color": "orange"},
    ax=axs[2]
)
axs[2].set_ylabel("count")
axs[2].set_title("Rental count with the windspeed Correlation")

plt.tight_layout()
st.pyplot(fig)
