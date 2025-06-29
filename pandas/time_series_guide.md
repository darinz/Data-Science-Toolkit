# Pandas Time Series Analysis: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Time Series](#introduction-to-time-series)
2. [Creating Time Series Data](#creating-time-series-data)
3. [Time Series Indexing](#time-series-indexing)
4. [Resampling and Frequency Conversion](#resampling-and-frequency-conversion)
5. [Rolling and Expanding Windows](#rolling-and-expanding-windows)
6. [Time Series Visualization](#time-series-visualization)
7. [Seasonal Decomposition](#seasonal-decomposition)
8. [Time Series Forecasting](#time-series-forecasting)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

## Introduction to Time Series

Time series data consists of observations collected at regular time intervals. Pandas provides powerful tools for:

- **Time series creation** and manipulation
- **Resampling** and frequency conversion
- **Rolling statistics** and moving averages
- **Seasonal decomposition** and trend analysis
- **Time series forecasting** and modeling

### Setup and Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
values = np.random.normal(100, 10, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 365) * 5

ts = pd.Series(values, index=dates)
print(f"Time series shape: {ts.shape}")
print(f"Date range: {ts.index.min()} to {ts.index.max()}")
print(f"Frequency: {ts.index.freq}")
```

## Creating Time Series Data

### 1. Basic Time Series Creation

```python
# From datetime index
dates = pd.date_range('2023-01-01', periods=365, freq='D')
daily_data = pd.Series(np.random.randn(365), index=dates)
print(f"Daily time series:\n{daily_data.head()}")

# From existing DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'value': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Set date as index
df_ts = df.set_index('date')
print(f"\nDataFrame with datetime index:\n{df_ts.head()}")

# Create time series from specific column
ts_from_df = df.set_index('date')['value']
print(f"\nTime series from DataFrame column:\n{ts_from_df.head()}")
```

### 2. Different Frequencies

```python
# Different time frequencies
frequencies = {
    'Hourly': pd.date_range('2023-01-01', periods=24, freq='H'),
    'Daily': pd.date_range('2023-01-01', periods=30, freq='D'),
    'Weekly': pd.date_range('2023-01-01', periods=12, freq='W'),
    'Monthly': pd.date_range('2023-01-01', periods=12, freq='M'),
    'Quarterly': pd.date_range('2023-01-01', periods=4, freq='Q'),
    'Yearly': pd.date_range('2023-01-01', periods=5, freq='Y')
}

for freq_name, dates in frequencies.items():
    ts_freq = pd.Series(np.random.randn(len(dates)), index=dates)
    print(f"\n{freq_name} time series:")
    print(f"Shape: {ts_freq.shape}")
    print(f"Frequency: {ts_freq.index.freq}")
    print(f"Sample: {ts_freq.head(3).values}")
```

### 3. Business Time Series

```python
# Business day frequency (excludes weekends)
business_days = pd.date_range('2023-01-01', periods=30, freq='B')
business_ts = pd.Series(np.random.randn(30), index=business_days)
print(f"Business day time series:\n{business_ts.head()}")

# Month end frequency
month_end = pd.date_range('2023-01-01', periods=12, freq='BM')
month_end_ts = pd.Series(np.random.randn(12), index=month_end)
print(f"\nMonth end time series:\n{month_end_ts.head()}")

# Quarter end frequency
quarter_end = pd.date_range('2023-01-01', periods=4, freq='BQ')
quarter_end_ts = pd.Series(np.random.randn(4), index=quarter_end)
print(f"\nQuarter end time series:\n{quarter_end_ts.head()}")
```

## Time Series Indexing

### 1. Basic Time Indexing

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts = pd.Series(np.random.randn(365), index=dates)

print(f"Time series:\n{ts.head()}")

# Date-based indexing
specific_date = ts['2023-01-15']
print(f"\nValue on 2023-01-15: {specific_date}")

# Date range indexing
date_range = ts['2023-01-01':'2023-01-10']
print(f"\nDate range:\n{date_range}")

# Year indexing
year_2023 = ts['2023']
print(f"\nYear 2023 data shape: {year_2023.shape}")

# Month indexing
january = ts['2023-01']
print(f"\nJanuary 2023 data shape: {january.shape}")
```

### 2. Advanced Time Indexing

```python
# Partial string indexing
jan_data = ts['2023-01']
print(f"January data:\n{jan_data.head()}")

# Time slicing
morning_data = ts['2023-01-01 00:00:00':'2023-01-01 12:00:00']
print(f"\nMorning data shape: {morning_data.shape}")

# Using truncate
truncated = ts.truncate(before='2023-06-01', after='2023-08-31')
print(f"\nTruncated data (June-August): {truncated.shape}")

# Boolean indexing with time conditions
recent_data = ts[ts.index > '2023-06-01']
print(f"\nRecent data (after June 1): {recent_data.shape}")
```

### 3. Time Series Selection Methods

```python
# First and last observations
first_obs = ts.first('30D')  # First 30 days
last_obs = ts.last('30D')    # Last 30 days

print(f"First 30 days:\n{first_obs.head()}")
print(f"\nLast 30 days:\n{last_obs.head()}")

# Between times
between_data = ts.between_time('00:00:00', '12:00:00')
print(f"\nData between 00:00 and 12:00: {between_data.shape}")

# At specific times
at_time_data = ts.at_time('09:00:00')
print(f"\nData at 09:00: {at_time_data.shape}")
```

## Resampling and Frequency Conversion

### 1. Downsampling (Higher to Lower Frequency)

```python
# Create hourly data
hourly_dates = pd.date_range('2023-01-01', periods=24*30, freq='H')
hourly_ts = pd.Series(np.random.randn(24*30), index=hourly_dates)

print(f"Hourly data shape: {hourly_ts.shape}")

# Resample to daily (mean)
daily_mean = hourly_ts.resample('D').mean()
print(f"\nDaily mean:\n{daily_mean.head()}")

# Resample to daily (sum)
daily_sum = hourly_ts.resample('D').sum()
print(f"\nDaily sum:\n{daily_sum.head()}")

# Resample to weekly
weekly_mean = hourly_ts.resample('W').mean()
print(f"\nWeekly mean:\n{weekly_mean.head()}")

# Resample to monthly
monthly_mean = hourly_ts.resample('M').mean()
print(f"\nMonthly mean:\n{monthly_mean.head()}")
```

### 2. Upsampling (Lower to Higher Frequency)

```python
# Create daily data
daily_dates = pd.date_range('2023-01-01', periods=30, freq='D')
daily_ts = pd.Series(np.random.randn(30), index=daily_dates)

print(f"Daily data:\n{daily_ts.head()}")

# Resample to hourly (forward fill)
hourly_ffill = daily_ts.resample('H').ffill()
print(f"\nHourly data (forward fill):\n{hourly_ffill.head(10)}")

# Resample to hourly (backward fill)
hourly_bfill = daily_ts.resample('H').bfill()
print(f"\nHourly data (backward fill):\n{hourly_bfill.head(10)}")

# Resample to hourly (interpolation)
hourly_interp = daily_ts.resample('H').interpolate()
print(f"\nHourly data (interpolation):\n{hourly_interp.head(10)}")
```

### 3. Custom Resampling

```python
# Custom aggregation functions
def custom_agg(x):
    return {
        'mean': x.mean(),
        'std': x.std(),
        'min': x.min(),
        'max': x.max(),
        'count': len(x)
    }

# Apply custom aggregation
custom_resampled = hourly_ts.resample('D').apply(custom_agg)
print(f"Custom resampled:\n{custom_resampled.head()}")

# Multiple aggregations
multi_agg = hourly_ts.resample('D').agg(['mean', 'std', 'min', 'max'])
print(f"\nMultiple aggregations:\n{multi_agg.head()}")

# Different aggregations for different columns
df_ts = pd.DataFrame({
    'value1': np.random.randn(100),
    'value2': np.random.randn(100)
}, index=pd.date_range('2023-01-01', periods=100, freq='H'))

custom_multi = df_ts.resample('D').agg({
    'value1': ['mean', 'std'],
    'value2': ['sum', 'count']
})
print(f"\nCustom multi-column aggregation:\n{custom_multi.head()}")
```

## Rolling and Expanding Windows

### 1. Rolling Windows

```python
# Create sample time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

print(f"Original time series:\n{ts.head()}")

# Rolling mean (7-day window)
rolling_mean = ts.rolling(window=7).mean()
print(f"\n7-day rolling mean:\n{rolling_mean.head(10)}")

# Rolling standard deviation
rolling_std = ts.rolling(window=7).std()
print(f"\n7-day rolling std:\n{rolling_std.head(10)}")

# Rolling statistics
rolling_stats = ts.rolling(window=7).agg(['mean', 'std', 'min', 'max'])
print(f"\nRolling statistics:\n{rolling_stats.head(10)}")

# Rolling with different window types
rolling_center = ts.rolling(window=7, center=True).mean()
rolling_min = ts.rolling(window=7, min_periods=3).mean()
print(f"\nCentered rolling mean:\n{rolling_center.head(10)}")
```

### 2. Expanding Windows

```python
# Expanding mean (all previous observations)
expanding_mean = ts.expanding().mean()
print(f"Expanding mean:\n{expanding_mean.head(10)}")

# Expanding standard deviation
expanding_std = ts.expanding().std()
print(f"\nExpanding std:\n{expanding_std.head(10)}")

# Expanding statistics
expanding_stats = ts.expanding().agg(['mean', 'std', 'min', 'max'])
print(f"\nExpanding statistics:\n{expanding_stats.head(10)}")

# Compare rolling vs expanding
comparison = pd.DataFrame({
    'Original': ts,
    'Rolling_Mean_7': ts.rolling(window=7).mean(),
    'Expanding_Mean': ts.expanding().mean()
})
print(f"\nComparison:\n{comparison.head(10)}")
```

### 3. Custom Rolling Functions

```python
# Custom rolling function
def rolling_range(x):
    return x.max() - x.min()

# Apply custom function
rolling_range_vals = ts.rolling(window=7).apply(rolling_range)
print(f"Rolling range:\n{rolling_range_vals.head(10)}")

# Rolling quantiles
rolling_median = ts.rolling(window=7).quantile(0.5)
rolling_q75 = ts.rolling(window=7).quantile(0.75)
print(f"\nRolling median:\n{rolling_median.head(10)}")
print(f"\nRolling 75th percentile:\n{rolling_q75.head(10)}")

# Rolling correlation (if we have two series)
ts2 = pd.Series(np.random.randn(100), index=dates)
rolling_corr = ts.rolling(window=7).corr(ts2)
print(f"\nRolling correlation:\n{rolling_corr.head(10)}")
```

## Time Series Visualization

### 1. Basic Time Series Plots

```python
# Create sample data with trend and seasonality
dates = pd.date_range('2023-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonality = 5 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 1, 365)
ts = pd.Series(trend + seasonality + noise, index=dates)

# Basic line plot
plt.figure(figsize=(12, 6))
ts.plot()
plt.title('Time Series with Trend and Seasonality')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Multiple time series
ts2 = pd.Series(trend + 2*seasonality + noise, index=dates)
ts3 = pd.Series(trend + 0.5*seasonality + noise, index=dates)

plt.figure(figsize=(12, 6))
ts.plot(label='Series 1')
ts2.plot(label='Series 2')
ts3.plot(label='Series 3')
plt.title('Multiple Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Rolling Statistics Visualization

```python
# Plot original data with rolling statistics
plt.figure(figsize=(12, 8))

# Original data
plt.subplot(2, 1, 1)
ts.plot(label='Original Data', alpha=0.7)
ts.rolling(window=30).mean().plot(label='30-day Rolling Mean', linewidth=2)
ts.rolling(window=30).std().plot(label='30-day Rolling Std', linewidth=2)
plt.title('Time Series with Rolling Statistics')
plt.legend()
plt.grid(True)

# Rolling statistics separately
plt.subplot(2, 1, 2)
ts.rolling(window=7).mean().plot(label='7-day Rolling Mean')
ts.rolling(window=30).mean().plot(label='30-day Rolling Mean')
ts.rolling(window=90).mean().plot(label='90-day Rolling Mean')
plt.title('Rolling Means with Different Windows')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. Seasonal and Trend Analysis

```python
# Seasonal subseries plot
monthly_data = ts.resample('M').mean()
monthly_data.index = monthly_data.index.month

plt.figure(figsize=(12, 6))
monthly_data.plot(kind='bar')
plt.title('Monthly Averages')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Box plot by month
ts_df = ts.reset_index()
ts_df['month'] = ts_df['index'].dt.month
ts_df['month_name'] = ts_df['index'].dt.strftime('%b')

plt.figure(figsize=(12, 6))
sns.boxplot(data=ts_df, x='month_name', y=0)
plt.title('Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()
```

## Seasonal Decomposition

### 1. Basic Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts, period=365, extrapolate_trend='freq')

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

plt.tight_layout()
plt.show()

# Print decomposition statistics
print("Decomposition Statistics:")
print(f"Trend range: {decomposition.trend.min():.2f} to {decomposition.trend.max():.2f}")
print(f"Seasonal range: {decomposition.seasonal.min():.2f} to {decomposition.seasonal.max():.2f}")
print(f"Residual std: {decomposition.resid.std():.2f}")
```

### 2. Multiplicative vs Additive Decomposition

```python
# Create data with multiplicative seasonality
multiplicative_ts = pd.Series(trend * (1 + 0.3 * seasonality) + noise, index=dates)

# Additive decomposition
additive_decomp = seasonal_decompose(ts, period=365, extrapolate_trend='freq')

# Multiplicative decomposition
multiplicative_decomp = seasonal_decompose(multiplicative_ts, period=365, 
                                         model='multiplicative', extrapolate_trend='freq')

# Compare
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

additive_decomp.seasonal.plot(ax=axes[0,0], title='Additive Seasonal')
multiplicative_decomp.seasonal.plot(ax=axes[0,1], title='Multiplicative Seasonal')
additive_decomp.resid.plot(ax=axes[1,0], title='Additive Residual')
multiplicative_decomp.resid.plot(ax=axes[1,1], title='Multiplicative Residual')

plt.tight_layout()
plt.show()
```

### 3. STL Decomposition

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust)
stl_decomp = STL(ts, period=365).fit()

# Plot STL decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

stl_decomp.observed.plot(ax=axes[0], title='Observed')
stl_decomp.trend.plot(ax=axes[1], title='Trend')
stl_decomp.seasonal.plot(ax=axes[2], title='Seasonal')
stl_decomp.resid.plot(ax=axes[3], title='Residual')

plt.tight_layout()
plt.show()

# Compare with classical decomposition
print("Decomposition Comparison:")
print(f"Classical residual std: {decomposition.resid.std():.4f}")
print(f"STL residual std: {stl_decomp.resid.std():.4f}")
```

## Time Series Forecasting

### 1. Simple Forecasting Methods

```python
# Moving average forecast
def moving_average_forecast(ts, window, periods_ahead):
    """Simple moving average forecast."""
    ma = ts.rolling(window=window).mean()
    last_ma = ma.iloc[-1]
    forecast = pd.Series([last_ma] * periods_ahead, 
                        index=pd.date_range(ts.index[-1] + pd.Timedelta(days=1), 
                                           periods=periods_ahead, freq='D'))
    return forecast

# Exponential smoothing forecast
def exponential_smoothing_forecast(ts, alpha, periods_ahead):
    """Simple exponential smoothing forecast."""
    smoothed = ts.ewm(alpha=alpha).mean()
    last_smoothed = smoothed.iloc[-1]
    forecast = pd.Series([last_smoothed] * periods_ahead,
                        index=pd.date_range(ts.index[-1] + pd.Timedelta(days=1),
                                           periods=periods_ahead, freq='D'))
    return forecast

# Generate forecasts
ma_forecast = moving_average_forecast(ts, window=30, periods_ahead=30)
es_forecast = exponential_smoothing_forecast(ts, alpha=0.3, periods_ahead=30)

# Plot forecasts
plt.figure(figsize=(12, 6))
ts.plot(label='Historical Data')
ma_forecast.plot(label='Moving Average Forecast', linestyle='--')
es_forecast.plot(label='Exponential Smoothing Forecast', linestyle='--')
plt.title('Time Series Forecasting')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. ARIMA Modeling

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Check stationarity
def check_stationarity(ts):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(ts.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    return result[1] < 0.05

# Check if series is stationary
is_stationary = check_stationarity(ts)
print(f"\nSeries is stationary: {is_stationary}")

# If not stationary, difference the series
if not is_stationary:
    ts_diff = ts.diff().dropna()
    is_stationary_diff = check_stationarity(ts_diff)
    print(f"After differencing, series is stationary: {is_stationary_diff}")

# Fit ARIMA model
try:
    # Use differenced series if original is not stationary
    model_series = ts_diff if not is_stationary else ts
    
    # Fit ARIMA(1,1,1) model
    model = ARIMA(model_series, order=(1, 1, 1))
    fitted_model = model.fit()
    
    print(f"\nARIMA Model Summary:")
    print(fitted_model.summary())
    
    # Generate forecast
    forecast = fitted_model.forecast(steps=30)
    print(f"\nForecast:\n{forecast.head()}")
    
except Exception as e:
    print(f"ARIMA modeling failed: {e}")
```

### 3. Seasonal ARIMA (SARIMA)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
try:
    # SARIMA(1,1,1)(1,1,1,12) for monthly data
    monthly_ts = ts.resample('M').mean()
    
    model = SARIMAX(monthly_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit(disp=False)
    
    print(f"SARIMA Model Summary:")
    print(fitted_model.summary())
    
    # Generate forecast
    forecast = fitted_model.forecast(steps=12)
    print(f"\n12-month forecast:\n{forecast}")
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    monthly_ts.plot(label='Historical Data')
    forecast.plot(label='SARIMA Forecast', linestyle='--')
    plt.title('SARIMA Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"SARIMA modeling failed: {e}")
```

## Best Practices

### 1. Data Quality Checks

```python
def validate_time_series(ts):
    """Validate time series data quality."""
    issues = []
    
    # Check for missing values
    missing_count = ts.isnull().sum()
    if missing_count > 0:
        issues.append(f"Missing values: {missing_count}")
    
    # Check for duplicate timestamps
    duplicate_timestamps = ts.index.duplicated().sum()
    if duplicate_timestamps > 0:
        issues.append(f"Duplicate timestamps: {duplicate_timestamps}")
    
    # Check for irregular frequency
    if ts.index.freq is None:
        issues.append("Irregular frequency detected")
    
    # Check for outliers
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((ts < Q1 - 1.5 * IQR) | (ts > Q3 + 1.5 * IQR)).sum()
    if outliers > 0:
        issues.append(f"Outliers detected: {outliers}")
    
    return issues

# Validate our time series
issues = validate_time_series(ts)
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("No data quality issues found")
```

### 2. Performance Optimization

```python
import time

def measure_performance(func, *args, **kwargs):
    """Measure function execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Compare different resampling methods
def resample_method_1(ts):
    return ts.resample('D').mean()

def resample_method_2(ts):
    return ts.groupby(ts.index.date).mean()

# Performance comparison
methods = [resample_method_1, resample_method_2]
method_names = ['resample()', 'groupby()']

for method, name in zip(methods, method_names):
    result, execution_time = measure_performance(method, ts)
    print(f"{name}: {execution_time:.6f} seconds")
```

### 3. Memory Management

```python
def optimize_time_series_memory(ts):
    """Optimize memory usage for time series."""
    initial_memory = ts.memory_usage(deep=True)
    
    # Optimize data type
    if ts.dtype == 'float64':
        ts_optimized = pd.to_numeric(ts, downcast='float')
    else:
        ts_optimized = ts
    
    final_memory = ts_optimized.memory_usage(deep=True)
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"Memory optimization:")
    print(f"Initial memory: {initial_memory / 1024:.2f} KB")
    print(f"Final memory: {final_memory / 1024:.2f} KB")
    print(f"Memory reduction: {memory_reduction:.1f}%")
    
    return ts_optimized

# Optimize memory
ts_optimized = optimize_time_series_memory(ts)
```

## Common Pitfalls

### 1. Time Zone Issues

```python
# Create time series with timezone
ts_tz = pd.Series(np.random.randn(100), 
                  index=pd.date_range('2023-01-01', periods=100, freq='D', tz='UTC'))

print(f"Time series with timezone:\n{ts_tz.head()}")

# Convert timezone
ts_est = ts_tz.tz_convert('US/Eastern')
print(f"\nConverted to EST:\n{ts_est.head()}")

# Remove timezone
ts_no_tz = ts_tz.tz_localize(None)
print(f"\nTimezone removed:\n{ts_no_tz.head()}")
```

### 2. Frequency Mismatches

```python
# Create irregular time series
irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-07', '2023-01-10'])
irregular_ts = pd.Series([1, 2, 3, 4], index=irregular_dates)

print(f"Irregular time series:\n{irregular_ts}")

# Infer frequency
inferred_freq = irregular_ts.index.inferred_freq
print(f"Inferred frequency: {inferred_freq}")

# Resample with forward fill
regular_ts = irregular_ts.resample('D').ffill()
print(f"\nRegularized time series:\n{regular_ts}")
```

### 3. Seasonality Detection

```python
def detect_seasonality(ts, max_period=50):
    """Detect seasonality using autocorrelation."""
    from statsmodels.tsa.stattools import acf
    
    # Calculate autocorrelation
    acf_values = acf(ts.dropna(), nlags=max_period)
    
    # Find peaks (potential seasonal periods)
    peaks = []
    for i in range(1, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            if acf_values[i] > 0.1:  # Threshold for significance
                peaks.append(i)
    
    return peaks

# Detect seasonality
seasonal_periods = detect_seasonality(ts)
print(f"Detected seasonal periods: {seasonal_periods}")

# Plot autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(12, 4))
plot_acf(ts.dropna(), lags=50)
plt.title('Autocorrelation Function')
plt.show()
```

## Summary

This guide covered comprehensive time series analysis with pandas:

1. **Time Series Creation**: Building time series from various sources
2. **Time Series Indexing**: Efficient data access and selection
3. **Resampling**: Frequency conversion and aggregation
4. **Rolling Windows**: Moving statistics and trends
5. **Visualization**: Time series plotting and analysis
6. **Seasonal Decomposition**: Trend, seasonal, and residual components
7. **Forecasting**: Simple and advanced forecasting methods
8. **Best Practices**: Data quality and performance optimization
9. **Common Pitfalls**: Timezone and frequency issues

### Key Takeaways

- **Choose appropriate frequencies** for your analysis
- **Handle missing data** carefully in time series
- **Use rolling windows** for trend analysis
- **Decompose seasonality** to understand patterns
- **Validate forecasts** with out-of-sample testing
- **Consider time zones** and frequency consistency

### Next Steps

- Practice with real-world time series data
- Explore advanced forecasting models
- Learn about multivariate time series
- Study time series clustering
- Master real-time data processing

### Additional Resources

- [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Statsmodels Time Series](https://www.statsmodels.org/stable/tsa.html)
- [Time Series Analysis in Python](https://machinelearningmastery.com/time-series-analysis-in-python/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)

---

**Ready to analyze time series data? Start exploring temporal patterns in your datasets!** 