#  F1 Race Predictor 2025 - Machine Learning Model 🏎

This is a *F1 preditions 2025* model repository. This project uses the FastF1 API data and F1 previous race results to predict outcomes for each race of the 2025 F1 season.

## Project Overview 📦
This project uses historical race data and qualifying lap times to predict the total race time for each F1 driver in the 2025 season. This repository contains a Linear Regression model built with scikit-learn to estimate total race times using:
- 2025 qualification session data.
- Historical race results for each driver.
- Feature engineering to convert lap times into usable input.
- Integrates weather forecast data from Visual Crossing API
- Outputs predicted race lap times per driver

## Dependencies 📊 
- fastf1
- numpy
- pandas
- scikit-learn
- matplotlib

## How it works 🏁 
This project uses machine learning to predict Formula 1 race lap times by analyzing multiple features such as qualifying times, weather conditions, and historical performance data. Here's how it works:
1. *Data collection*: The project collects data from various sources, including Formula 1 qualifying and lap times, weather data and driver and team statistics
2. *Preprocessing & Feature Engineering*: Converts lap times, normalizes driver names, and structures race data.
3. *Model Trainning*:A Gradient Boosting Regressor model is trained using these features to predict the race lap time for each driver.
4. *Prediction*: After training the model, it is used to predict the race lap times for drivers based on the input features.
5. *Evaluation*: The model's accuracy is evaluated using Mean Absolute Error (MAE), which measures the difference between predicted lap times and actual lap times. A lower MAE indicates a more accurate model.
6. *Results*: The model outputs a predicted lap time for each driver in a given race, which is then used to simulate potential race outcomes.

## Ideas for Extension 🧠
1. Add real-time telemetry data
2. Train with more historical races and conditions
3. Support race team strategy prediction or race result classification
4. Visualize the prediction results using matplotlib 
   

 





