This part of the project documentation focuses on an **understanding-oriented**
approach. You'll get a chance to read about the background of the project, as well as
reasoning about how it was implemented.

# Background

This project was initially proposed as a solution for the Machine Learning in Finance,
in the Post Graduation Program of Operational Research, at Aeronautics Institute of
Technology (ITA).

The development of the project was based book
[Advances in Financial Machine Learning](
https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086),
by Marcos Lopez de Prado. The book presents a methodology for developing a
Meta-Labeling Application for financial time series, with the purpose of predicting
the direction of the next price.

The course covers a wide range of topics on Finance, to be able to understand all
the process on the Investment Management, from a strategy design up to the deployment
of the solution.

In this context, the project was developed with the purpose of applying the
knowledge acquired during the course, in addition of providing a solution for the
Meta-Labeling Application.

# Solution

The solution is a Meta-Labeling Application for financial time series, with the
purpose of predicting the direction of the next price. It is recommended to read
the article [finance-machine-learning-meta-labeling-application](
/materials/lima2023finance.pdf) to understand the application on financial
time series, and to understand the concepts of Meta-Labeling.

# Application

The solution was build to be used as a library, by importing and implementing the
functions developed in the project. Financial Analysts can use the solution to
develop their own Meta-Labeling Application.

The methods were developed to be used in a financial time series, with the purpose
of predicting the direction of the next price. Although the implementation simplicity,
the solution requires some decision taking by the user, such as the choice of the
financial asset, the time frame, the number of days to be predicted, the number of
days to be used in the feature engineering, the number of days to be used in the
meta-labeling, the number of days to be used in the training and the number of days
to be used in the test.

# Development

For the development of this solution, the following steps were followed:

1. Data collection
2. Data cleaning
3. Feature engineering
4. Model training

The knowledge required for the development of the solution were based on the
following topics:

- Financial Technical Analysis
- Machine Learning
- Python programming language

And it is adjustable to any financial time series, as long as the user
understands the concepts of financial technical analysis and machine learning.

# Support materials

The solution was developed in Python, using the following libraries:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

# Collaboration

The project is open source, and it can be reused and modified by anyone. Feel free
to contribute to the project, by forking it and submitting a pull request.

# References

[De Prado, M. L. (2018). Advances in financial machine learning. John Wiley & Sons.](
https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

[Sefidian, A. M. (2021) Labeling financial data for Machine Learning. Sefidian Academy](
    https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/)
