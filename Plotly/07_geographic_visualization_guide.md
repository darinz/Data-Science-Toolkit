# Plotly Geographic Visualization Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Maps](https://img.shields.io/badge/Geographic-Maps-green.svg)](https://plotly.com/python/maps/)

A comprehensive guide to creating interactive geographic visualizations with Plotly, including choropleth maps, scattergeo plots, map projections, and advanced mapping techniques for spatial data analysis.

## Table of Contents

1. [Introduction to Geographic Visualization](#introduction-to-geographic-visualization)
2. [Choropleth Maps](#choropleth-maps)
3. [Scattergeo Plots](#scattergeo-plots)
4. [Map Projections](#map-projections)
5. [Geographic Data Handling](#geographic-data-handling)
6. [Interactive Maps](#interactive-maps)
7. [Advanced Mapping Techniques](#advanced-mapping-techniques)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)
10. [Common Applications](#common-applications)

## Introduction to Geographic Visualization

Geographic visualizations help you understand spatial patterns, distributions, and relationships in your data by displaying information on maps.

### Why Geographic Visualization?

- **Spatial Patterns** - Identify geographic trends and clusters
- **Regional Analysis** - Compare data across different regions
- **Location-based Insights** - Understand data in geographic context
- **Interactive Exploration** - Zoom, pan, and explore geographic data

### Basic Setup

```python
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Sample geographic data
countries = ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina']
values = [100, 85, 70, 60, 45]
```

## Choropleth Maps

### Basic Choropleth Map

```python
# Create basic choropleth map
fig = px.choropleth(
    locations=countries,
    locationmode='country names',
    color=values,
    color_continuous_scale='Viridis',
    title="Basic Choropleth Map"
)

fig.show()
```

### Choropleth with Custom Data

```python
# Create DataFrame for choropleth
df = pd.DataFrame({
    'country': ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Peru'],
    'population': [331002651, 37742154, 128932753, 212559417, 45195774, 19116201, 32971854],
    'gdp': [21433200, 1647000, 1086000, 1833000, 445500, 282300, 228990]
})

fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    hover_name='country',
    color_continuous_scale='Plasma',
    title="Population by Country",
    labels={'population': 'Population'}
)

fig.show()
```

### Custom Choropleth Styling

```python
# Create choropleth with custom styling
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='gdp',
    hover_name='country',
    hover_data=['population', 'gdp'],
    color_continuous_scale='RdBu',
    range_color=[0, df['gdp'].max()],
    title="GDP by Country"
)

# Customize layout
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        projection_type='equirectangular'
    )
)

fig.show()
```

### Regional Choropleth Maps

```python
# US States choropleth
us_states = ['California', 'Texas', 'Florida', 'New York', 'Illinois']
us_values = [39512223, 28995881, 21477737, 19453561, 12671821]

fig = px.choropleth(
    locations=us_states,
    locationmode='USA-states',
    color=us_values,
    color_continuous_scale='Viridis',
    title="US States Population",
    scope='usa'
)

fig.show()
```

## Scattergeo Plots

### Basic Scattergeo Plot

```python
# Create scattergeo plot
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
latitudes = [40.7128, 34.0522, 41.8781, 29.7604, 33.4484]
longitudes = [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
populations = [8336817, 3979576, 2693976, 2320268, 1680992]

fig = px.scatter_geo(
    lat=latitudes,
    lon=longitudes,
    size=populations,
    hover_name=cities,
    title="US Cities Population"
)

fig.show()
```

### Scattergeo with Custom Data

```python
# Create DataFrame for scattergeo
cities_df = pd.DataFrame({
    'city': cities,
    'lat': latitudes,
    'lon': longitudes,
    'population': populations,
    'region': ['Northeast', 'West', 'Midwest', 'South', 'West']
})

fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    hover_data=['population'],
    title="US Cities by Region",
    scope='usa'
)

fig.show()
```

### Custom Scattergeo Styling

```python
# Create scattergeo with custom styling
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='population',
    hover_name='city',
    color_continuous_scale='Viridis',
    size_max=30,
    title="US Cities with Custom Styling"
)

# Customize layout
fig.update_layout(
    geo=dict(
        scope='usa',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        showlakes=True,
        lakecolor='blue',
        showrivers=True,
        rivercolor='blue',
        projection_scale=1.5,
        center=dict(lat=39.8283, lon=-98.5795)
    )
)

fig.show()
```

## Map Projections

### Different Projection Types

```python
# Create choropleth with different projections
projections = ['equirectangular', 'mercator', 'orthographic', 'natural earth']

for proj in projections:
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='population',
        title=f"World Population - {proj.title()} Projection"
    )
    
    fig.update_layout(
        geo=dict(
            projection_type=proj,
            showframe=False,
            showcoastlines=True,
            coastlinecolor='black'
        )
    )
    
    fig.show()
```

### Custom Projection Parameters

```python
# Create map with custom projection parameters
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    title="Custom Projection Parameters"
)

fig.update_layout(
    geo=dict(
        projection_type='orthographic',
        projection_rotation=dict(lon=0, lat=0, roll=0),
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    )
)

fig.show()
```

### Regional Projections

```python
# Create regional maps with appropriate projections
regions = {
    'usa': {'scope': 'usa', 'projection': 'albers usa'},
    'europe': {'scope': 'europe', 'projection': 'mercator'},
    'asia': {'scope': 'asia', 'projection': 'mercator'},
    'africa': {'scope': 'africa', 'projection': 'mercator'}
}

for region, config in regions.items():
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color='population',
        title=f"{region.title()} Population"
    )
    
    fig.update_layout(
        geo=dict(
            scope=config['scope'],
            projection_type=config['projection'],
            showframe=False,
            showcoastlines=True,
            coastlinecolor='black'
        )
    )
    
    fig.show()
```

## Geographic Data Handling

### Working with Coordinates

```python
# Handle different coordinate formats
def convert_coordinates(lat, lon, from_format='decimal', to_format='decimal'):
    """Convert between coordinate formats"""
    if from_format == 'decimal' and to_format == 'decimal':
        return lat, lon
    elif from_format == 'dms' and to_format == 'decimal':
        # Convert DMS to decimal
        lat_decimal = lat[0] + lat[1]/60 + lat[2]/3600
        lon_decimal = lon[0] + lon[1]/60 + lon[2]/3600
        return lat_decimal, lon_decimal
    # Add more conversion functions as needed

# Example usage
lat_dms = (40, 42, 46)  # Degrees, minutes, seconds
lon_dms = (-74, 0, 22)
lat_dec, lon_dec = convert_coordinates(lat_dms, lon_dms, 'dms', 'decimal')
```

### Geographic Data Validation

```python
# Validate geographic coordinates
def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates"""
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} is out of range [-90, 90]")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} is out of range [-180, 180]")
    return True

# Example usage
try:
    validate_coordinates(40.7128, -74.0060)
    print("Coordinates are valid")
except ValueError as e:
    print(f"Invalid coordinates: {e}")
```

### Geographic Data Cleaning

```python
# Clean geographic data
def clean_geographic_data(df, lat_col='lat', lon_col='lon'):
    """Clean geographic data by removing invalid coordinates"""
    # Remove rows with missing coordinates
    df_clean = df.dropna(subset=[lat_col, lon_col])
    
    # Remove rows with invalid coordinates
    valid_mask = (
        (df_clean[lat_col] >= -90) & (df_clean[lat_col] <= 90) &
        (df_clean[lon_col] >= -180) & (df_clean[lon_col] <= 180)
    )
    
    return df_clean[valid_mask]

# Example usage
sample_data = pd.DataFrame({
    'city': ['New York', 'Invalid', 'Los Angeles'],
    'lat': [40.7128, 200, 34.0522],  # Invalid latitude
    'lon': [-74.0060, -74.0060, -118.2437]
})

clean_data = clean_geographic_data(sample_data)
print(clean_data)
```

## Interactive Maps

### Hover Information

```python
# Create map with rich hover information
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    hover_data=['population', 'region'],
    title="Interactive US Cities Map"
)

# Customize hover template
fig.update_traces(
    hovertemplate="<b>%{hover_name}</b><br>" +
                  "Population: %{customdata[0]:,}<br>" +
                  "Region: %{customdata[1]}<br>" +
                  "<extra></extra>"
)

fig.show()
```

### Click Events

```python
# Create map with click events
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    title="Clickable US Cities Map"
)

# In a real application, you would add callbacks here
# For example, in a Dash app:
"""
@app.callback(
    Output('city-info', 'children'),
    Input('map', 'clickData')
)
def display_city_info(clickData):
    if clickData is None:
        return "Click on a city to see information"
    return f"Selected: {clickData['points'][0]['hover_name']}"
"""

fig.show()
```

### Zoom and Pan Controls

```python
# Create map with custom zoom and pan controls
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    title="US Cities with Zoom Controls"
)

fig.update_layout(
    geo=dict(
        scope='usa',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        projection_scale=2,  # Initial zoom level
        center=dict(lat=39.8283, lon=-98.5795),  # Center of US
        lonaxis=dict(range=[-125, -65]),  # Longitude range
        lataxis=dict(range=[25, 50])      # Latitude range
    )
)

fig.show()
```

## Advanced Mapping Techniques

### Multiple Map Layers

```python
# Create map with multiple layers
fig = go.Figure()

# Add choropleth layer
fig.add_trace(go.Choropleth(
    locations=df['country'],
    locationmode='country names',
    z=df['population'],
    colorscale='Viridis',
    name='Population',
    showscale=True
))

# Add scattergeo layer
fig.add_trace(go.Scattergeo(
    lat=cities_df['lat'],
    lon=cities_df['lon'],
    mode='markers',
    marker=dict(
        size=cities_df['population']/100000,
        color='red',
        opacity=0.7
    ),
    name='Cities',
    text=cities_df['city'],
    hovertemplate="<b>%{text}</b><br>" +
                  "Population: %{marker.size:,.0f}<br>" +
                  "<extra></extra>"
))

fig.update_layout(
    title="Multi-layer Geographic Visualization",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    )
)

fig.show()
```

### Custom Map Styling

```python
# Create map with custom styling
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    title="Custom Styled Map"
)

fig.update_layout(
    geo=dict(
        scope='usa',
        showframe=True,
        framecolor='black',
        framewidth=2,
        showland=True,
        landcolor='rgb(243, 243, 243)',
        showocean=True,
        oceancolor='rgb(204, 229, 255)',
        showlakes=True,
        lakecolor='rgb(255, 255, 255)',
        showrivers=True,
        rivercolor='rgb(255, 255, 255)',
        showcoastlines=True,
        coastlinecolor='rgb(80, 80, 80)',
        coastlinewidth=1,
        projection_type='albers usa',
        projection_scale=1.5,
        center=dict(lat=39.8283, lon=-98.5795),
        lonaxis=dict(range=[-125, -65]),
        lataxis=dict(range=[25, 50])
    )
)

fig.show()
```

### Animated Maps

```python
# Create animated geographic visualization
# Sample time series data
years = [2010, 2015, 2020]
populations_2010 = [100, 85, 70, 60, 45]
populations_2015 = [110, 90, 75, 65, 50]
populations_2020 = [120, 95, 80, 70, 55]

fig = px.choropleth(
    locations=countries,
    locationmode='country names',
    color=populations_2020,
    animation_frame=0,  # Placeholder for animation
    color_continuous_scale='Viridis',
    title="Population Growth Over Time"
)

# In a real implementation, you would create frames for each year
# This is a simplified example

fig.show()
```

## Performance Optimization

### Large Dataset Handling

```python
# Optimize maps for large datasets
def create_optimized_map(df, max_points=1000):
    """Create optimized map for large datasets"""
    if len(df) > max_points:
        # Downsample data
        df = df.sample(n=max_points, random_state=42)
    
    fig = px.scatter_geo(
        df,
        lat='lat',
        lon='lon',
        size='population',
        color='region',
        hover_name='city',
        title="Optimized Map for Large Dataset"
    )
    
    return fig

# Example usage
large_cities_df = pd.DataFrame({
    'city': [f'City_{i}' for i in range(10000)],
    'lat': np.random.uniform(25, 50, 10000),
    'lon': np.random.uniform(-125, -65, 10000),
    'population': np.random.randint(10000, 1000000, 10000),
    'region': np.random.choice(['Northeast', 'West', 'Midwest', 'South'], 10000)
})

fig = create_optimized_map(large_cities_df)
fig.show()
```

### Efficient Rendering

```python
# Create efficient map rendering
def create_efficient_choropleth(df, color_column, title):
    """Create efficient choropleth map"""
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color=color_column,
        color_continuous_scale='Viridis',
        title=title
    )
    
    # Optimize layout
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='black',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='lightblue'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

# Example usage
fig = create_efficient_choropleth(df, 'population', 'Efficient Choropleth')
fig.show()
```

## Best Practices

### 1. Color and Contrast

```python
# Use appropriate color scales for geographic data
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    color_continuous_scale='Viridis',  # Good for sequential data
    title="Population with Good Color Contrast"
)

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    )
)

fig.show()
```

### 2. Map Projection Selection

```python
# Choose appropriate projections for different regions
projection_guide = {
    'world': 'equirectangular',
    'usa': 'albers usa',
    'europe': 'mercator',
    'asia': 'mercator',
    'africa': 'mercator',
    'polar': 'stereographic'
}

# Example for US data
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    title="Appropriate Projection for US Data"
)

fig.update_layout(
    geo=dict(
        scope='usa',
        projection_type='albers usa',  # Good for US
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black'
    )
)

fig.show()
```

### 3. Interactive Features

```python
# Add helpful interactive features
fig = px.scatter_geo(
    cities_df,
    lat='lat',
    lon='lon',
    size='population',
    color='region',
    hover_name='city',
    title="Interactive Map with Features"
)

fig.update_layout(
    geo=dict(
        scope='usa',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        projection_scale=1.5,
        center=dict(lat=39.8283, lon=-98.5795)
    )
)

fig.show()
```

### 4. Accessibility

```python
# Make maps accessible
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    title="Accessible Geographic Map"
)

fig.update_layout(
    geo=dict(
        showframe=True,
        framecolor='black',
        framewidth=2,
        showcoastlines=True,
        coastlinecolor='black',
        coastlinewidth=2,
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    ),
    font=dict(size=14)  # Larger fonts for accessibility
)

fig.show()
```

## Common Applications

### 1. Population Analysis

```python
# Population density analysis
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='population',
    color_continuous_scale='Viridis',
    title="World Population Distribution",
    labels={'population': 'Population'}
)

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    )
)

fig.show()
```

### 2. Economic Data Visualization

```python
# GDP visualization
fig = px.choropleth(
    df,
    locations='country',
    locationmode='country names',
    color='gdp',
    color_continuous_scale='RdBu',
    title="GDP by Country",
    labels={'gdp': 'GDP (millions USD)'}
)

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue'
    )
)

fig.show()
```

### 3. Climate and Weather Data

```python
# Temperature data visualization
temperature_data = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'lat': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
    'temperature': [45, 75, 35, 70, 85]
})

fig = px.scatter_geo(
    temperature_data,
    lat='lat',
    lon='lon',
    size='temperature',
    color='temperature',
    hover_name='city',
    color_continuous_scale='RdBu_r',
    title="Temperature by City",
    scope='usa'
)

fig.show()
```

## Summary

Plotly geographic visualization provides powerful tools for spatial data analysis:

- **Choropleth Maps**: Visualize data by regions and countries
- **Scattergeo Plots**: Display point data on maps
- **Map Projections**: Choose appropriate projections for different regions
- **Interactive Features**: Zoom, pan, hover, and click interactions
- **Advanced Techniques**: Multi-layer maps and custom styling
- **Performance**: Optimization for large datasets
- **Best Practices**: Color selection, projection choice, and accessibility

Master these geographic visualization techniques to create compelling, interactive maps that effectively communicate spatial relationships in your data.

## Next Steps

- Explore [Plotly Maps](https://plotly.com/python/maps/) for more examples
- Learn [Graph Objects](https://plotly.com/python/graph-objects/) for advanced customization
- Study [Geographic Data](https://plotly.com/python/choropleth-maps/) handling
- Practice [Interactive Features](https://plotly.com/python/interactive-plots/) for maps

---

**Happy Mapping!** 🗺️✨ 