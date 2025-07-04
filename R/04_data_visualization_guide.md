# R Data Visualization Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Intermediate-orange.svg)](./)
[![Category](https://img.shields.io/badge/Category-Data%20Visualization-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)
[![ggplot2](https://img.shields.io/badge/ggplot2-3.4.0+-green.svg)](https://ggplot2.tidyverse.org/)
[![plotly](https://img.shields.io/badge/plotly-4.10.0+-green.svg)](https://plotly.com/r/)

This guide covers data visualization in R, including base plotting, ggplot2, and advanced visualization techniques. Visualization helps you understand and communicate your data.

## Table of Contents
1. [Introduction](#introduction)
2. [Base Plotting](#base-plotting)
3. [ggplot2 Package](#ggplot2-package)
4. [Advanced Visualization](#advanced-visualization)
5. [Practice Exercises](#practice-exercises)
6. [Where to Learn More](#where-to-learn-more)

---

## Introduction
R offers several ways to create plots and charts. Start with base plotting for quick visuals, use ggplot2 for publication-quality graphics, and try advanced tools for interactivity.

## Base Plotting
- Built-in plotting functions: `plot()`, `hist()`, `boxplot()`, etc.
- Use for quick, simple visualizations.

```r
x <- 1:10
y <- x^2
plot(x, y, type="b", main="Base Plot Example")
hist(rnorm(100), main="Histogram Example")
```

## ggplot2 Package
- A powerful and flexible system for creating graphics.
- Use for layered, customizable, and publication-quality plots.
- Based on the Grammar of Graphics.

```r
library(ggplot2)
df <- data.frame(x=1:10, y=(1:10)^2)
ggplot(df, aes(x, y)) + geom_point() + geom_line() + ggtitle("ggplot2 Example")
```

## Advanced Visualization
- Interactive plots: `plotly`, `shiny`
- Specialized plots: `lattice`, `corrplot`, `heatmap`
- Use for dashboards, web apps, or specialized analysis.

```r
# Example with plotly
library(plotly)
p <- ggplot(df, aes(x, y)) + geom_point()
ggplotly(p)
```

---

## Practice Exercises
1. Create a scatter plot of two numeric vectors using base R.
2. Make a histogram of 100 random normal values.
3. Use ggplot2 to plot a line chart of y = x^2 for x from 1 to 10.
4. Try making an interactive plot with plotly.

## Where to Learn More
- [R for Data Science: Data Visualization](https://r4ds.hadley.nz/data-visualize.html)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [R Graph Gallery](https://r-graph-gallery.com/) 