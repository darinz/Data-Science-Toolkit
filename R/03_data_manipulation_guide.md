# R Data Manipulation Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Intermediate-orange.svg)](./)
[![Category](https://img.shields.io/badge/Category-Data%20Manipulation-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)
[![dplyr](https://img.shields.io/badge/dplyr-1.1.0+-green.svg)](https://dplyr.tidyverse.org/)
[![tidyr](https://img.shields.io/badge/tidyr-1.3.0+-green.svg)](https://tidyr.tidyverse.org/)

This guide covers essential data manipulation techniques in R, including subsetting, filtering, and using popular packages like dplyr, tidyr, and data.table.

## Table of Contents
1. [Introduction](#introduction)
2. [Subsetting and Filtering](#subsetting-and-filtering)
3. [dplyr Package](#dplyr-package)
4. [tidyr Package](#tidyr-package)
5. [data.table Package](#datatable-package)
6. [Practice Exercises](#practice-exercises)
7. [Where to Learn More](#where-to-learn-more)

---

## Introduction
Manipulating data is a core skill in R. You'll often need to select, filter, or reshape data before analysis. R provides both base functions and powerful packages for these tasks.

## Subsetting and Filtering
- Use `[]` for subsetting vectors, matrices, and data frames.
- Logical conditions can filter rows in data frames.
- Use for quick, simple data selection.

```r
vec <- c(10, 20, 30, 40)
vec[vec > 20] # returns 30, 40

df <- data.frame(Name=c("A", "B"), Age=c(25, 30))
df[df$Age > 25, ] # returns row with Age 30
```

## dplyr Package
- Provides a grammar for data manipulation.
- Use for readable, chainable data wrangling (especially with data frames).
- Common functions: `filter()`, `select()`, `mutate()`, `summarise()`, `arrange()`.

```r
library(dplyr)
df %>% filter(Age > 25) %>% select(Name)
```

## tidyr Package
- Used for tidying data: reshaping, pivoting, and cleaning up messy datasets.
- Use when you need to convert between wide and long formats.
- Common functions: `pivot_longer()`, `pivot_wider()`.

```r
library(tidyr)
wide_df <- data.frame(id=1:2, val1=c(10,20), val2=c(30,40))
long_df <- pivot_longer(wide_df, cols=starts_with("val"), names_to="variable", values_to="value")
```

## data.table Package
- High-performance data manipulation, especially for large datasets.
- Syntax is concise and fast for big data tasks.

```r
library(data.table)
dt <- data.table(Name=c("A", "B"), Age=c(25, 30))
dt[Age > 25, .(Name)]
```

---

## Practice Exercises
1. Subset a vector to only include values greater than 5.
2. Filter a data frame to show only rows where a column is equal to a specific value.
3. Use dplyr to select only the "Name" column from a data frame.
4. Use tidyr to convert a wide data frame to long format.
5. Use data.table to filter rows in a table where a value is above a threshold.

## Where to Learn More
- [R for Data Science: Data Transformation](https://r4ds.hadley.nz/transform.html)
- [dplyr Documentation](https://dplyr.tidyverse.org/)
- [tidyr Documentation](https://tidyr.tidyverse.org/)
- [data.table Documentation](https://rdatatable.gitlab.io/data.table/)

Mastering these tools will make your data wrangling in R efficient and effective. 