import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate synthetic climate data using NumPy
np.random.seed(100)  # Set seed for reproducibility
dates = pd.date_range(start='2024-01-01', periods=366, freq='D')  # Daily dates for a leap year
temperature = np.random.normal(28, 5, 366)  # Normally distributed temperatures (mean=28°C, std=5)
rainfall = np.random.poisson(10, size=366)  # Poisson-distributed rainfall (mean=10mm)
wind_speed = np.random.uniform(5, 25, 366)  # Uniformly distributed wind speeds between 5–25 km/h

# 2. Create a DataFrame from the generated data
climate_df = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Rainfall': rainfall,
    'Wind_speed': wind_speed
})

# 3. Data cleaning: introduce and handle missing values
climate_df.loc[10:15, 'Rainfall'] = np.nan  # Intentionally add NaNs to simulate missing data
climate_df["Rainfall"].fillna(climate_df["Rainfall"].mean(), inplace=True)  # Fill NaNs with mean rainfall

# 4. Data conversion and new column creation
climate_df['Temperature'] = np.round(climate_df['Temperature'], 1)  # Round temperatures to 1 decimal
climate_df['Month'] = climate_df['Date'].dt.month_name()  # Extract full month names

# 5. Calculate monthly statistics: average temperature and total rainfall
monthly_stats = climate_df.groupby('Month').agg(
    Avg_Temp=('Temperature', 'mean'),
    Total_Rainfall=('Rainfall', 'sum')
).reset_index()

# 6. Line plot of daily temperatures and 7-day rolling average
plt.figure(figsize=(12, 6))
plt.plot(climate_df['Date'], climate_df['Temperature'], label='Daily temperature', color='red', alpha=0.5)
plt.plot(climate_df['Date'], climate_df['Temperature'].rolling(7).mean(), color='black', label='Weekly average')
plt.title('2024 Annual Temperature Trend', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.savefig('../Result/temperature_trend.png')  # Save the figure
plt.close()

# 7. Bar graph of total monthly rainfall
# Define correct order for months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Set categorical order for sorting
monthly_stats['Month'] = pd.Categorical(
    monthly_stats['Month'],
    categories=month_order,
    ordered=True
)

# Sort by month order
monthly_stats = monthly_stats.sort_values('Month')

# Plot bar graph
plt.figure(figsize=(12, 6))
plt.bar(monthly_stats['Month'], monthly_stats['Total_Rainfall'], color='blue', alpha=0.7, edgecolor='black')
plt.title('Monthly Rainfall in 2024', fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('Total Rainfall (mm)', fontsize=14)
plt.savefig('../result/rainfall_distribution.png')  # Save the figure
plt.close()

# 8. Scatter plot: Temperature vs Wind Speed colored by Rainfall
plt.scatter(climate_df['Temperature'], climate_df['Wind_speed'], alpha=0.7, c=climate_df['Rainfall'], cmap='viridis')
plt.colorbar(label='Rainfall (mm)')  # Color bar indicating rainfall levels
plt.title('Wind Speed VS Temperature')
plt.ylabel('Wind Speed (km/h)')
plt.xlabel('Temperature (°C)')
plt.savefig("../result/wind_speed_scatter.png")  # Save the figure
plt.close()

# 9. Save cleaned and processed data to CSV
climate_df.to_csv('../data/processed_data.csv', index=False)
