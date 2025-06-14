# Stock Price Prediction Project 

## Project Overview

1. ### **Introduction** 🙋🏻‍♂️
Welcome to the Stock Price Prediction project documentation. This project focuses on predicting the stock price of "Mastercard" using the Prophet module, a forecasting tool developed by Facebook. Stock price prediction is a complex task that involves analyzing historical data, identifying patterns, and making predictions for future trends. This documentation provides a comprehensive guide to understanding the project, its components, and how to replicate or extend the work.

2. ### **Objectives** 🎯
The primary objectives of this project are as follows:

- Utilize historical stock price data for Mastercard to train a predictive model.
- Implement the Prophet module for time series forecasting.
- Evaluate the model's performance in predicting future stock prices.
- Conduct an experiment by omitting the last 90 days of historical data to assess the model's predictive accuracy.

3. ### **Technologies Used** 👨🏻‍💻
The project makes use of the following technologies:

- Python: The primary programming language used for data processing, analysis, and model implementation.
- Prophet Module: Developed by Facebook, Prophet is an open-source forecasting tool designed for time series data.
- pandas_datareader: Utilized to collect historical stock price data from 'Stooq'.
- Streamlit: Used for creating an interactive web application.

4. ### **Project Workflow** 👷🏻‍♂️⚒
The project follows a structured workflow:

- **Data Collection**: Historical stock price data for Mastercard is collected from Stooq using the pandas_datareader. Data is collected from 2015-01-01 onwards.

- **Data Preprocessing**: Only the relevant columns ('Date' and 'Close' Price) are extracted for the training data.

- **Model Training**: The Prophet module is used to train the predictive model on the preprocessed historical data.

- **Prediction**: The trained model is used to make predictions for future stock prices.

- **Experiment**: The model's predictive accuracy is tested by omitting the last 90 days of historical data, allowing for a direct comparison between actual and predicted results.

- **Visualization**: Results are visualized using graphs and charts to provide a clear understanding of the model's predictions and how they compare to actual stock prices.

5. ### **Web Application** 📺
A Stock Price Prediction Web Application is developed using Streamlit, providing the following features:

- Prediction for the next 4 years.
- Generation of a data table containing predicted values, future - dates, upper and lower price predictions, etc.
- Prediction chart showing both actual values till date and future predicted values.
- Forecast components showing predicted overall, weekly, and yearly trends of the stock.
- Link for the web application: https://machine-learning-lgg5gapgnvlynvoph7ezf5.streamlit.app/
