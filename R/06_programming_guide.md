# R Programming Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Intermediate-orange.svg)](./)
[![Category](https://img.shields.io/badge/Category-Programming-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)
[![Functions](https://img.shields.io/badge/Functions-Built--in-green.svg)](https://www.r-project.org/)

This guide covers programming concepts in R, including functions, control flow, the apply family, and package management. Programming lets you automate tasks and build reproducible analyses.

## Table of Contents
1. [Introduction](#introduction)
2. [Functions](#functions)
3. [Control Flow](#control-flow)
4. [Apply Family](#apply-family)
5. [Packages](#packages)
6. [Practice Exercises](#practice-exercises)
7. [Where to Learn More](#where-to-learn-more)

---

## Introduction
Learning to program in R helps you write reusable code, automate analyses, and work efficiently with data.

## Functions
- Use functions to organize and reuse code.
- Defined with `function()`.

```r
add <- function(a, b) {
  return(a + b)
}
add(2, 3)
```

## Control Flow
- Use `if`, `else`, and loops to control the flow of your program.
- Conditional statements: `if`, `else`, `else if`
- Loops: `for`, `while`, `repeat`

```r
x <- 5
if (x > 0) {
  print("Positive")
} else {
  print("Non-positive")
}

for (i in 1:3) {
  print(i)
}
```

## Apply Family
- Use `apply()`, `lapply()`, `sapply()`, `tapply()` to apply functions to data structures efficiently.

```r
mat <- matrix(1:6, nrow=2)
apply(mat, 1, sum)
l <- list(a=1:3, b=4:6)
lapply(l, mean)
sapply(l, mean)
```

## Packages
- Use packages to extend R's functionality.
- Install: `install.packages("package_name")`
- Load: `library(package_name)`

```r
install.packages("ggplot2")
library(ggplot2)
```

---

## Practice Exercises
1. Write a function that multiplies two numbers.
2. Use a for loop to print numbers 1 to 5.
3. Use `if` to check if a variable is positive or negative.
4. Use `lapply()` to calculate the mean of each element in a list.
5. Install and load the `dplyr` package.

## Where to Learn More
- [R for Data Science: Programming](https://r4ds.hadley.nz/program-intro.html)
- [Advanced R (free book)](https://adv-r.hadley.nz/)
- [RStudio Education](https://education.rstudio.com/learn/) 