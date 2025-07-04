# R Statistical Analysis Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Intermediate-orange.svg)](./)
[![Category](https://img.shields.io/badge/Category-Statistical%20Analysis-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)
[![Statistics](https://img.shields.io/badge/Statistics-Built--in-green.svg)](https://www.r-project.org/)

This guide covers statistical analysis in R, including descriptive statistics, hypothesis testing, and regression. R makes it easy to analyze and interpret data.

## Table of Contents
1. [Introduction](#introduction)
2. [Descriptive Statistics](#descriptive-statistics)
3. [Hypothesis Testing](#hypothesis-testing)
4. [Regression Analysis](#regression-analysis)
5. [Practice Exercises](#practice-exercises)
6. [Where to Learn More](#where-to-learn-more)

---

## Introduction
Statistical analysis helps you summarize, test, and model your data. R provides built-in functions and packages for a wide range of statistical tasks.

## Descriptive Statistics
- Use to summarize and describe your data.
- Functions: `mean()`, `median()`, `sd()`, `summary()`, `table()`

```r
x <- c(1, 2, 3, 4, 5)
mean(x)
median(x)
sd(x)
summary(x)
table(x)
```

## Hypothesis Testing
- Use to test assumptions or compare groups.
- t-test: `t.test()`
- Chi-squared test: `chisq.test()`
- ANOVA: `aov()`

```r
# t-test example
group1 <- c(5, 6, 7)
group2 <- c(8, 9, 10)
t.test(group1, group2)

# Chi-squared test example
tab <- table(c("A", "B", "A", "B", "A"))
chisq.test(tab)
```

## Regression Analysis
- Use to model relationships between variables.
- Linear regression: `lm()`
- Summary: `summary()`

```r
df <- data.frame(x=1:10, y=2*(1:10) + rnorm(10))
model <- lm(y ~ x, data=df)
summary(model)
```

---

## Practice Exercises
1. Calculate the mean and standard deviation of a numeric vector.
2. Perform a t-test comparing two groups of numbers.
3. Run a linear regression on a data frame with x and y columns.
4. Use `summary()` to get a quick overview of a dataset.

## Where to Learn More
- [R for Data Science: Model](https://r4ds.hadley.nz/model.html)
- [CRAN Task View: Statistics](https://cran.r-project.org/web/views/Statistics.html)
- [Quick-R: Statistical Analysis](https://www.statmethods.net/)

R provides a rich set of tools for statistical analysis. 