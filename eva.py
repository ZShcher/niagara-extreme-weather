import pandas as pd
import numpy as np 
from scipy.stats import genextreme
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r'C:\Users\zahar\Desktop\Hackathon Project\multiple_stations.csv')
df.columns = df.columns.str.strip() # Remove leading/trailing whitespace from column names

# Rename columns for easier access
df.rename(columns={
    'LOCAL_YEAR': 'year',
    'LOCAL_MONTH': 'month',
    'LOCAL_DAY': 'day',
    'MAX_TEMPERATURE': 'temperature_max',
    'MIN_TEMPERATURE': 'temperature_min',
    'TOTAL_RAIN': 'rain',
    'TOTAL_SNOW': 'snow'
}, inplace=True)

# Convert to numeric and track issues
# Save original row index for debugging
df['row_index'] = df.index
for col in ['year', 'month', 'day']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop invalid entries
df.dropna(subset=['year', 'month', 'day', 'temperature_max', 'temperature_min', 'rain', 'snow'], inplace=True)
df[['year', 'month', 'day']] = df[['year', 'month', 'day']].astype(int)

# Parse dates
bad_dates = []
dates = []
# Try constructing datetime and catch individual row errors
for i, row in df.iterrows():
    try:
        dates.append(pd.Timestamp(year=row['year'], month=row['month'], day=row['day']))
    except Exception as e:
        bad_dates.append((row['row_index'], row['year'], row['month'], row['day'], str(e)))
if bad_dates:
    for info in bad_dates:
        print(f"Row {info[0]} → ({info[1]}-{info[2]}-{info[3]}): {info[4]}")
df = df.drop(index=[r[0] for r in bad_dates]) # Drop rows with bad dates
df['date'] = pd.to_datetime(df[['year', 'month', 'day']]) # Create a date column from year, month, day
df.sort_values('date', inplace=True) # Sort by date

# Filter for seasonal daily extremes 
df_summer = df[df['month'].isin([6, 7, 8])]  # June–August for max temps
df_winter = df[df['month'].isin([12, 1, 2])] # Dec–Feb for min temps

# Extract data (daily, non-zero where needed)
max_data = df_summer['temperature_max'].dropna().values
min_data = df_winter['temperature_min'].dropna().values
rain_data = df[df['rain'] > 0]['rain'].dropna().values
snow_data = df[df['snow'] > 0]['snow'].dropna().values

# Fit GEV
c_max, loc_max, scale_max = genextreme.fit(max_data)
c_min, loc_min, scale_min = genextreme.fit(-min_data)
c_rain, loc_rain, scale_rain = genextreme.fit(rain_data)
c_snow, loc_snow, scale_snow = genextreme.fit(snow_data)

# Return periods
T_years = [10, 50, 100, 1000]

return_levels_max = [genextreme.ppf(1 - 1/t, c_max, loc=loc_max, scale=scale_max) for t in T_years]
return_levels_min = [-genextreme.ppf(1 - 1/t, c_min, loc=loc_min, scale=scale_min) for t in T_years]
return_levels_rain = [genextreme.ppf(1 - 1/t, c_rain, loc=loc_rain, scale=scale_rain) for t in T_years]
return_levels_snow = [genextreme.ppf(1 - 1/t, c_snow, loc=loc_snow, scale=scale_snow) for t in T_years]

# Output
print("Return levels for daily max temperatures (June–Aug):")
for T, level in zip(T_years, return_levels_max):
    print(f"{T}-year return level: {level:.2f} °C")

print("\nReturn levels for daily min temperatures (Dec–Feb):")
for T, level in zip(T_years, return_levels_min):
    print(f"{T}-year return level: {level:.2f} °C")

print("\nReturn levels for daily rain (rain > 0):")
for T, level in zip(T_years, return_levels_rain):
    print(f"{T}-year return level: {level:.2f} mm")

print("\nReturn levels for daily snow (snow > 0):")
for T, level in zip(T_years, return_levels_snow):
    print(f"{T}-year return level: {level:.2f} mm")

# Plotting
x = np.linspace(min(max_data), max(max_data), 100)
pdf = genextreme.pdf(x, c_max, loc=loc_max, scale=scale_max)
plt.figure(figsize=(10, 6))
plt.hist(max_data, bins=30, density=True, alpha=0.5)
plt.plot(x, pdf, 'r-', label='Max Temp Fit')
plt.title('GEV Fit: Daily Max Temperature (Apr–Aug)')
plt.xlabel('Daily Maximum Temperature (°C)')
plt.ylabel('Density')
plt.legend(); plt.grid(True); plt.show()

x = np.linspace(min(min_data), max(min_data), 100)
pdf = genextreme.pdf(-x, c_min, loc=loc_min, scale=scale_min)
plt.figure(figsize=(10, 6))
plt.hist(min_data, bins=30, density=True, alpha=0.5)
plt.plot(x, pdf, 'b-', label='Min Temp Fit')
plt.title('GEV Fit: Daily Min Temperature (Dec–Feb)')
plt.xlabel('Daily Minimum Temperature (°C)')
plt.ylabel('Density')
plt.legend(); plt.grid(True); plt.show()

x = np.linspace(min(rain_data), max(rain_data), 100)
pdf = genextreme.pdf(x, c_rain, loc=loc_rain, scale=scale_rain)
plt.figure(figsize=(10, 6))
plt.hist(rain_data, bins=30, density=True, alpha=0.5)
plt.plot(x, pdf, 'g-', label='Rain Fit')
plt.title('GEV Fit: Daily Rain (Rain > 0)')
plt.xlabel('Daily Rain (mm)')
plt.ylabel('Density')
plt.legend(); plt.grid(True); plt.show()

x = np.linspace(min(snow_data), max(snow_data), 100)
pdf = genextreme.pdf(x, c_snow, loc=loc_snow, scale=scale_snow)
plt.figure(figsize=(10, 6))
plt.hist(snow_data, bins=30, density=True, alpha=0.5)
plt.plot(x, pdf, 'purple', label='Snow Fit')
plt.title('GEV Fit: Daily Snow (Snow > 0)')
plt.xlabel('Daily Snow (mm)')
plt.ylabel('Density')
plt.legend(); plt.grid(True); plt.show()
