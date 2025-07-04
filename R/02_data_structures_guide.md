# R Data Structures Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Beginner-green.svg)](./)
[![Category](https://img.shields.io/badge/Category-Data%20Structures-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)

This guide covers the core data structures in R: vectors, lists, matrices, data frames, and factors. Understanding these is essential for effective R programming.

## Table of Contents
1. [Introduction](#introduction)
2. [Vectors](#vectors)
3. [Lists](#lists)
4. [Matrices](#matrices)
5. [Data Frames](#data-frames)
6. [Factors](#factors)
7. [Practice Exercises](#practice-exercises)
8. [Where to Learn More](#where-to-learn-more)

---

## Introduction
R provides several built-in data structures for storing and manipulating data. Choosing the right one depends on your data and analysis needs.

## Vectors
- The most basic data structure in R.
- Homogeneous: all elements must be of the same type (numeric, character, logical, etc.).
- Use vectors for simple, one-dimensional data.
- Created with `c()` function.

```r
num_vec <- c(1, 2, 3, 4)
char_vec <- c("a", "b", "c")
log_vec <- c(TRUE, FALSE, TRUE)
```

## Lists
- Can contain elements of different types (numbers, strings, vectors, even other lists).
- Useful for grouping related but different types of data.
- Created with `list()`.

```r
my_list <- list(1, "a", TRUE, c(1,2,3))
```

## Matrices
- Two-dimensional, homogeneous data structure (all elements must be the same type).
- Use for numeric or character data arranged in rows and columns.
- Created with `matrix()`.

```r
mat <- matrix(1:6, nrow=2, ncol=3)
```

## Data Frames
- Two-dimensional, heterogeneous data structure (columns can be different types).
- Most common structure for datasets (like spreadsheets).
- Created with `data.frame()`.

```r
df <- data.frame(Name=c("A", "B"), Age=c(25, 30))
```

## Factors
- Used for categorical data (e.g., colors, gender, status).
- Store data as levels (categories) rather than raw values.
- Created with `factor()`.

```r
colors <- factor(c("red", "blue", "red", "green"))
levels(colors)
```

---

## Practice Exercises
1. Create a numeric vector with the numbers 1 to 5.
2. Make a list containing a number, a string, and a logical value.
3. Create a 3x3 matrix of numbers from 1 to 9.
4. Make a data frame with two columns: one for names, one for ages.
5. Create a factor for the categories "apple", "banana", "apple", "orange".

## Where to Learn More
- [R for Data Science: Data Structures](https://r4ds.hadley.nz/data-structures.html)
- [R Vectors Tutorial](https://www.datamentor.io/r-programming/vector/)
- [R Data Frames](https://www.datamentor.io/r-programming/data-frame/) 