import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# import csv file as a dataframe
df = pd.read_csv('./data/Spain.csv')

#restrict the data to the dateime (local) and the Price columns
df = df[['Datetime (Local)', 'Price (EUR/MWhe)']]

# change the date format
df['Datetime (Local)'] = pd.to_datetime(df['Datetime (Local)'])
# make a smoothed price column by taking the rolling mean of the price column
df['Smoothed Price (EUR/MWhe)'] = df['Price (EUR/MWhe)'].rolling(window=48).mean()

df.plot(x='Datetime (Local)', y='Smoothed Price (EUR/MWhe)')

#save the plot as a png file
plt.savefig('./plots/price_plot.png')



# Filter the DataFrame to include only the last week of data
last_date = df['Datetime (Local)'].max()
one_week_ago = last_date - pd.DateOffset(weeks=1)
last_week_df = df[df['Datetime (Local)'] >= one_week_ago]
last_week_df.plot(x='Datetime (Local)', y='Price (EUR/MWhe)')



# Plotting
fig, ax = plt.subplots()

# Plot the smoothed price
ax.plot(last_week_df['Datetime (Local)'], last_week_df['Price (EUR/MWhe)'], label='Smoothed Price')

# Add background color gradient for day and night
for _, row in last_week_df.iterrows():
    if row['Datetime (Local)'].hour >= 6 and row['Datetime (Local)'].hour < 22:
        ax.axvspan(row['Datetime (Local)'], row['Datetime (Local)'] + pd.Timedelta(hours=1), color='yellow', alpha=0.1)
    else:

        ax.axvspan(row['Datetime (Local)'], row['Datetime (Local)'] + pd.Timedelta(hours=1), color='blue', alpha=0.1)

# Formatting the x-axis
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xlabel('Datetime (Local)')
plt.ylabel('Price (EUR/MWhe)')
plt.title('Price with Day and Night Indication')
plt.legend()

plt.savefig('./plots/price_day_night.png')

# Filter the DataFrame to include only the last week of data
last_week_df = df[df['Datetime (Local)'] >= one_week_ago].copy()

# Calculate the Fourier Transform
price_data = df['Price (EUR/MWhe)'].values
fft_result = np.fft.fft(price_data)
fft_freq = np.fft.fftfreq(len(price_data))

# Plotting the Fourier Transform
plt.figure(figsize=(10, 6))
plt.plot(fft_freq, np.abs(fft_result))
plt.yscale('log')
plt.xlabel('Frequency')
plt.ylabel('Amplitude (log scale)')
plt.title('Fourier Transform of Price Time Series')

plt.savefig('./plots/fourier_transform.png')