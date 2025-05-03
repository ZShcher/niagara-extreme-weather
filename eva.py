import pandas as pd
import numpy as np 
from scipy.stats import genextreme
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r'C:\Users\zahar\Desktop\Hackathon Project\1942-2025.csv')
df.columns = df.columns.str.strip() # Remove leading/trailing whitespace from column names

# Rename columns to standard names if necessary
df.rename(columns={
    'LOCAL_YEAR': 'year',
    'LOCAL_MONTH': 'month',
    'LOCAL_DAY': 'day',
    'MAX_TEMPERATURE': 'temperature_max',
    'MIN_TEMPERATURE': 'temperature_min'
}, inplace=True)

# Convert to numeric and track issues
df['row_index'] = df.index  # Save original row index for debugging
for col in ['year', 'month', 'day']:
    df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, set invalid parsing as NaN

# Find rows with missing date parts
invalid_date_rows = df[df[['year', 'month', 'day']].isnull().any(axis=1)]
if not invalid_date_rows.empty:
    print("Rows with invalid date parts:") # Print rows with invalid date parts
    print(invalid_date_rows[['row_index', 'year', 'month', 'day']]) 

# Drop rows with invalid date or temperature (NaN values)
df.dropna(subset=['year', 'month', 'day', 'temperature_max', 'temperature_min'], inplace=True)

# Convert to int
df[['year', 'month', 'day']] = df[['year', 'month', 'day']].astype(int)

# Try constructing datetime and catch individual row errors
bad_dates = []
dates = []
for i, row in df.iterrows():
    try:
        dates.append(pd.Timestamp(year=row['year'], month=row['month'], day=row['day'])) # Create date object 
    except Exception as e:
        bad_dates.append((row['row_index'], row['year'], row['month'], row['day'], str(e))) # Store errors for debugging

if bad_dates: # Print rows where date conversion failed
    print("Rows where date conversion failed:")
    for info in bad_dates:
        print(f"Row {info[0]} → ({info[1]}-{info[2]}-{info[3]}): {info[4]}")

# Create 'date' column from valid dates
df = df.drop(index=[r[0] for r in bad_dates]) # Drop rows with bad dates
df['date'] = pd.to_datetime(df[['year', 'month', 'day']]) # Create a date column from year, month, day

# Sort by date
df.sort_values('date', inplace=True)

# Compute block maxima/minima: max and min temperatures per month
monthly_max = df.groupby(['year', 'month'])['temperature_max'].max().reset_index()
monthly_min = df.groupby(['year', 'month'])['temperature_min'].min().reset_index()

# Prepare data for EVA
max_data = monthly_max['temperature_max'].dropna().values # Drop NaN values for max temperatures
min_data = monthly_min['temperature_min'].dropna().values # Drop NaN values for min temperatures

# Fit the GEV distribution for max temps
c_max, loc_max, scale_max = genextreme.fit(max_data)

# Fit the GEV distribution for min temps (use -1 * values to treat minima as extremes)
c_min, loc_min, scale_min = genextreme.fit(-min_data)

# Compute return levels
T_years = [10, 50, 100, 1000] # Return periods in years
T_months = [12 * T for T in T_years]

return_levels_max = [genextreme.ppf(1 - 1/t, c_max, loc=loc_max, scale=scale_max) for t in T_months]
return_levels_min = [-genextreme.ppf(1 - 1/t, c_min, loc=loc_min, scale=scale_min) for t in T_months]

# Print results
print("Return levels for maximum temperatures:")
for T, level in zip(T_years, return_levels_max):
    print(f"{T}-year return level: {level:.2f} °C")

print("\nReturn levels for minimum temperatures:")
for T, level in zip(T_years, return_levels_min): 
    print(f"{T}-year return level: {level:.2f} °C")

# Plot for max temperatures
x_max = np.linspace(min(max_data), max(max_data), 100)
pdf_max = genextreme.pdf(x_max, c_max, loc=loc_max, scale=scale_max)

plt.figure(figsize=(10, 6))
plt.hist(max_data, bins=30, density=True, alpha=0.5, label='Observed Monthly Maxima')
plt.plot(x_max, pdf_max, 'r-', label='Fitted GEV PDF')
plt.title('Block Maxima and Fitted GEV Distribution (Max Temps)')
plt.xlabel('Monthly Maximum Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Plot for min temperatures
x_min = np.linspace(min(min_data), max(min_data), 100)
pdf_min = genextreme.pdf(-x_min, c_min, loc=loc_min, scale=scale_min)

plt.figure(figsize=(10, 6))
plt.hist(min_data, bins=30, density=True, alpha=0.5, label='Observed Monthly Minima')
plt.plot(x_min, pdf_min, 'b-', label='Fitted GEV PDF')
plt.title('Block Minima and Fitted GEV Distribution (Min Temps)')
plt.xlabel('Monthly Minimum Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
