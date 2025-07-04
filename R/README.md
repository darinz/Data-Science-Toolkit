# R Programming

[![R](https://img.shields.io/badge/R-4.3.0+-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Last Updated](https://img.shields.io/date/1685577600)](../../)
[![Documentation](https://img.shields.io/badge/docs-complete-brightgreen)](./)
[![CRAN](https://img.shields.io/badge/CRAN-available-green.svg)](https://cran.r-project.org/)
[![RStudio](https://img.shields.io/badge/RStudio-compatible-orange.svg)](https://posit.co/download/rstudio-desktop/)

This directory contains guides, tutorials, and resources for learning and using the R programming language for data analysis, statistics, and visualization.

## Quick Start

1. **Install R** ([CRAN Download](https://cran.r-project.org/))
2. **(Recommended) Install RStudio** ([RStudio Download](https://posit.co/download/rstudio-desktop/))
3. **Open RStudio or R GUI**
4. **Try running code in the Console:**

```r
print("Hello, R world!")
```

5. **Create a new script (.R file), paste code, and click 'Run'**

---

## Installation & Setup

### 1. Install R
- Download and install R from the official site: [https://cran.r-project.org/](https://cran.r-project.org/)
- Follow the instructions for your operating system (Windows, macOS, Linux).

### 2. (Recommended) Install RStudio
- Download and install RStudio (an IDE for R): [https://posit.co/download/rstudio-desktop/](https://posit.co/download/rstudio-desktop/)

### 3. Install Recommended Packages
Open R or RStudio and run:

```r
install.packages(c("tidyverse", "data.table", "ggplot2", "dplyr", "tidyr", "readr", "shiny", "knitr", "rmarkdown", "Rcpp"))
```

### 4. Verify Installation
Test your setup by running:

```r
library(ggplot2)
qplot(mpg, hp, data = mtcars)
```

---

## Table of Contents
- [R Basics Guide](01_r_basics_guide.md)
- [Data Structures Guide](02_data_structures_guide.md)
- [Data Manipulation Guide](03_data_manipulation_guide.md)
- [Data Visualization Guide](04_data_visualization_guide.md)
- [Statistical Analysis Guide](05_statistical_analysis_guide.md)
- [Programming Guide](06_programming_guide.md)
- [Advanced Features Guide](07_advanced_features_guide.md)
- [Real-World Applications Guide](08_real_world_applications_guide.md)

---

## Where to Learn More
- [R for Data Science (free book)](https://r4ds.hadley.nz/)
- [RStudio Education](https://education.rstudio.com/learn/)
- [CRAN R Manuals](https://cran.r-project.org/manuals.html)
- [Swirl: Learn R in R](https://swirlstats.com/)
- [RStudio Cheatsheets](https://posit.co/resources/cheatsheets/) 