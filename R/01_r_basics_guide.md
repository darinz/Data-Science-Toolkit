# R Basics Guide

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![Level](https://img.shields.io/badge/Level-Beginner-green.svg)](./)
[![Category](https://img.shields.io/badge/Category-Basics-yellow.svg)](./)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](./)

Welcome to the R Basics Guide! This guide introduces the fundamental concepts of the R programming language, including syntax, data types, variables, and operators.

## Table of Contents
1. [Introduction to R](#introduction-to-r)
2. [How to Run R Code](#how-to-run-r-code)
3. [R Syntax](#r-syntax)
4. [Data Types in R](#data-types-in-r)
5. [Variables](#variables)
6. [Operators](#operators)
7. [Comments](#comments)
8. [Getting Help](#getting-help)
9. [Practice Exercises](#practice-exercises)
10. [Where to Learn More](#where-to-learn-more)

---

## Introduction to R
R is a powerful language and environment for statistical computing and graphics. It is widely used among statisticians and data scientists for data analysis and visualization.

## How to Run R Code
- **R Console:** Open R or RStudio and type code directly into the console.
- **Script File:** Create a new file with a `.R` extension, write your code, and click 'Run' in RStudio or use `source('filename.R')` in the console.
- **RMarkdown:** Use `.Rmd` files to mix code and text for reports.

Example (in console or script):
```r
print("Hello, R world!")
```

## R Syntax
- R is case-sensitive.
- Statements are typically written one per line.
- The assignment operator is `<-`, but `=` can also be used.
- Functions are called with parentheses, e.g., `mean(x)`.

```r
# Example of assignment and function call
x <- 5
print(x)
```

## Data Types in R
R supports several basic data types:
- **Numeric**: Numbers with or without decimals (e.g., `3`, `4.5`)
- **Integer**: Whole numbers (e.g., `2L`)
- **Character**: Text strings (e.g., 'hello')
- **Logical**: Boolean values (`TRUE`, `FALSE`)
- **Complex**: Complex numbers (e.g., `1+2i`)

```r
num <- 3.14
int <- 2L
char <- "R language"
logi <- TRUE
comp <- 1+2i
```

## Variables
- Variable names can contain letters, numbers, `.` and `_`, but cannot start with a number.
- Use `<-` or `=` for assignment.

```r
my_var <- 10
anotherVar = 20
```

## Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `^`, `%%`, `%/%`
- **Relational**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical**: `&`, `|`, `!`, `&&`, `||`

```r
a <- 5
b <- 2
sum <- a + b
prod <- a * b
is_equal <- a == b
```

## Comments
- Use `#` for single-line comments.

```r
# This is a comment
```

## Getting Help
- Use `?function_name` or `help(function_name)` to access documentation.

```r
?mean
help("mean")
```

---

## Practice Exercises
1. Assign the value 42 to a variable called `answer` and print it.
2. Create a numeric variable, a character variable, and a logical variable.
3. Use arithmetic operators to calculate `7 * 8` and store the result in a variable.
4. Use a relational operator to check if 10 is greater than 5.
5. Add a comment to your code explaining what it does.

## Where to Learn More
- [R for Data Science (free book)](https://r4ds.hadley.nz/)
- [RStudio Education](https://education.rstudio.com/learn/)
- [CRAN R Manuals](https://cran.r-project.org/manuals.html)
- [Swirl: Learn R in R](https://swirlstats.com/)

Happy coding in R! 