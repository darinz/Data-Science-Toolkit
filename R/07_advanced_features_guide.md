# Advanced R Features Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Advanced-red.svg)](./)
[![Category](https://img.shields.io/badge/Category-Advanced%20Features-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)
[![Rcpp](https://img.shields.io/badge/Rcpp-1.0.0+-green.svg)](https://rcpp.org/)
[![S3/S4](https://img.shields.io/badge/S3/S4-Built--in-green.svg)](https://www.r-project.org/)

This guide covers advanced features in R, including S3/S4 object systems, environments, functional programming, and Rcpp integration. These tools help you write more powerful and efficient R code.

## Table of Contents
1. [Introduction](#introduction)
2. [S3 and S4 Object Systems](#s3-and-s4-object-systems)
3. [Environments](#environments)
4. [Functional Programming](#functional-programming)
5. [Rcpp Integration](#rcpp-integration)
6. [Practice Exercises](#practice-exercises)
7. [Where to Learn More](#where-to-learn-more)

---

## Introduction
Once you're comfortable with the basics, R offers advanced features for custom data structures, performance, and functional programming.

## S3 and S4 Object Systems
- S3: Informal, generic functions and classes. Use for simple custom objects.
- S4: Formal, with explicit class and method definitions. Use for complex, structured objects.

```r
# S3 example
person <- list(name="Alice", age=30)
class(person) <- "Person"
print.Person <- function(x) {
  cat(x$name, "is", x$age, "years old\n")
}
person

# S4 example
setClass("Person", slots=list(name="character", age="numeric"))
alice <- new("Person", name="Alice", age=30)
```

## Environments
- Used for variable scoping and namespaces. Useful for advanced programming and package development.

```r
e <- new.env()
e$x <- 42
env <- environment()
```

## Functional Programming
- Functions as first-class objects. Use `Map()`, `Reduce()`, `Filter()` for concise, powerful code.

```r
add_one <- function(x) x + 1
Map(add_one, 1:5)
Reduce(`+`, 1:5)
```

## Rcpp Integration
- Use C++ code in R for performance-critical tasks. Great for speeding up slow R code.

```r
# Install Rcpp
install.packages("Rcpp")
library(Rcpp)

# Example C++ function
cppFunction('int add(int x, int y) { return x + y; }')
add(2, 3)
```

---

## Practice Exercises
1. Create a simple S3 object and print it.
2. Create an S4 class and instantiate an object.
3. Use an environment to store a variable and retrieve it.
4. Use `Map()` to add 1 to each element of a vector.
5. (Optional) Try using Rcpp to write a simple C++ function in R.

## Where to Learn More
- [Advanced R (free book)](https://adv-r.hadley.nz/)
- [Rcpp Documentation](https://rcpp.org/)
- [R Environments](https://adv-r.hadley.nz/environments.html) 