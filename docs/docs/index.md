# student_needs documentation!

## Description

A regression and grouping of different student financial needs.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Open data sets
market_2024 = pd.read_csv("student_needs_repo/data/processed/2024marketdata.csv")

student_spending = pd.read_csv("student_needs_repo/data/processed/student_spending.csv")

monthly_expense = pd.read_csv("student_needs_repo/data/processed/student_monthly_expense.csv")