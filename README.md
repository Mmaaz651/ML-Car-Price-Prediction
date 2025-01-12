# Car Price Prediction using Machine Learning

This project demonstrates the use of machine learning algorithms to predict the price of used cars based on various features such as fuel type, seller type, transmission, and other car-related attributes. The project uses **Linear Regression** and **Lasso Regression** models to make predictions and evaluate model performance.

## Description

The goal of this project is to predict the price of cars based on the following features:

- **Fuel_Type**: Type of fuel the car uses (Petrol, Diesel, CNG)
- **Seller_Type**: The type of seller (Dealer, Individual)
- **Transmission**: Transmission type (Manual, Automatic)
- **Car specifications**: Other car-related features like age, mileage, engine type, etc.

### Steps:

1. **Data Collection & Preprocessing**:
   - The dataset is loaded from a CSV file.
   - The data is cleaned by handling missing values and encoding categorical variables (`Fuel_Type`, `Seller_Type`, and `Transmission`) into numerical values.

2. **Model Training**:
   - Two machine learning models are implemented:
     - **Linear Regression**: A basic model to predict car prices based on the input features.
     - **Lasso Regression**: A regularized version of linear regression that helps prevent overfitting.

3. **Model Evaluation**:
   - The models are trained on the training dataset and evaluated on both training and test datasets.
   - **R-squared error** is used to evaluate the performance of the models.
   - Visualizations (scatter plots) compare actual and predicted prices.

4. **Result Visualization**:
   - Scatter plots are used to visualize how close the predicted car prices are to the actual prices for both training and test datasets.

## Requirements

To run this project, you will need:

- Python 3.x
- **pandas**: For data manipulation and preprocessing.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For implementing machine learning algorithms.
