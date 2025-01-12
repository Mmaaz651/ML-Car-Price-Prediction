# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn import metrics
#
# # Data collection and processing
#
# #loading the data from csv file to pandas data frame
# car_dataset=pd.read_csv('car data.csv')
#
# #inspecting first five rows of the data frame
# print(car_dataset.head())
#
# #checking the number of rows and columns
# print(car_dataset.shape)
#
# # getting some information about the dataset
# print(car_dataset.info)
#
# #checking the number of missing values
# print(car_dataset.isnull().sum)
#
# # checking the distribution of categorical data
# print(car_dataset.Fuel_Type.value_counts())
# print(car_dataset.Seller_Type.value_counts())
# print(car_dataset.Transmission.value_counts())
#
# # encoding the categorical data
#
# #encoding "Fuel_Type" column
# car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
#
# #encoding "Seller_Type" column
# car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#
# #encoding "Transmission" column
# car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
#
# print(car_dataset.head())
#
#
# #Splitting the data into Training data and Test data
#
# x=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
# y=car_dataset['Selling_Price']
#
# print(x)
# print(y)
#
# #Splitting Training and Test Data
#
# x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1, random_state=2)
#
# #Model Training
# #Linear Regression
# #loading the linear regression model
#
# # Linear Regression
#
# lin_reg_model=LinearRegression()
# lin_reg_model.fit(x_train,y_train)   #fit is used to train
#
# #model evaluation
# #prediction on training data
# training_data_prediction= lin_reg_model.predict(x_train)
#
#
# #R squared error
# error_score=metrics.r2_score(y_train, training_data_prediction)
# print(f'R Squared Error: {error_score}')
#
# #Visualize the actual prices and predicted prices
# plt.scatter(y_train, training_data_prediction)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual Prices vs Predicted Prices')
# #plt.show()
#
#
# #prediction on Test data
# test_data_prediction= lin_reg_model.predict(x_test)
#
# #R squared error
# error_score=metrics.r2_score(y_test, test_data_prediction)
# print(f'R Squared Error: {error_score}')
#
#
# plt.scatter(y_test, test_data_prediction)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual Prices vs Predicted Prices')
# plt.show()
#
#
# # Lasso Regression
#
# lass_reg_model=Lasso()
# lass_reg_model.fit(x_train,y_train)   #fit is used to train
#
# #model evaluation
# #prediction on training data
# training_data_prediction= lass_reg_model.predict(x_train)
#
#
# #R squared error
# error_score=metrics.r2_score(y_train, training_data_prediction)
# print(f'R Squared Error: {error_score}')
#
# #Visualize the actual prices and predicted prices
# plt.scatter(y_train, training_data_prediction)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual Prices vs Predicted Prices')
# #plt.show()
#
#
# #prediction on Test data
# test_data_prediction= lass_reg_model.predict(x_test)
#
# #R squared error
# error_score=metrics.r2_score(y_test, test_data_prediction)
# print(f'R Squared Error: {error_score}')
#
#
# plt.scatter(y_test, test_data_prediction)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual Prices vs Predicted Prices')
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# Data collection and preprocessing

# Loading the car dataset from a CSV file into a pandas DataFrame
car_dataset = pd.read_csv('car data.csv')

# Displaying the first five rows of the dataset to understand its structure
print(car_dataset.head())

# Checking the number of rows and columns in the dataset
print(car_dataset.shape)

# Getting basic information about the dataset such as data types and memory usage
print(car_dataset.info())

# Checking for any missing values in the dataset
print(car_dataset.isnull().sum())

# Examining the distribution of the categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# Encoding categorical variables into numeric values

# Encoding "Fuel_Type" column: Petrol -> 0, Diesel -> 1, CNG -> 2
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Encoding "Seller_Type" column: Dealer -> 0, Individual -> 1
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# Encoding "Transmission" column: Manual -> 0, Automatic -> 1
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Displaying the dataset again to verify the changes after encoding
print(car_dataset.head())

# Splitting the data into features (X) and target variable (y)

# Dropping 'Car_Name' and 'Selling_Price' columns from the feature set
x = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)

# Target variable is 'Selling_Price'
y = car_dataset['Selling_Price']

print(x)
print(y)

# Splitting the dataset into training and test sets

# Using a 90-10 split for training and testing, with random_state set for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Model Training

# Training a Linear Regression model

# Creating an instance of the Linear Regression model
lin_reg_model = LinearRegression()

# Fitting the model with the training data
lin_reg_model.fit(x_train, y_train)

# Evaluating the model on the training data

# Predicting selling prices on the training data
training_data_prediction = lin_reg_model.predict(x_train)

# Calculating the R-squared error for the training data prediction
error_score = metrics.r2_score(y_train, training_data_prediction)
print(f'R Squared Error (Training Data): {error_score}')

# Visualizing the actual prices vs predicted prices for training data
plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Data: Actual vs Predicted Prices')
#plt.show()

# Predicting selling prices on the test data
test_data_prediction = lin_reg_model.predict(x_test)

# Calculating the R-squared error for the test data prediction
error_score = metrics.r2_score(y_test, test_data_prediction)
print(f'R Squared Error (Test Data): {error_score}')

# Visualizing the actual prices vs predicted prices for test data
plt.scatter(y_test, test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Test Data: Actual vs Predicted Prices')
plt.show()

# Model Training with Lasso Regression

# Creating an instance of the Lasso Regression model
lass_reg_model = Lasso()

# Fitting the model with the training data
lass_reg_model.fit(x_train, y_train)

# Evaluating the model on the training data

# Predicting selling prices on the training data using Lasso Regression
training_data_prediction = lass_reg_model.predict(x_train)

# Calculating the R-squared error for the training data prediction with Lasso
error_score = metrics.r2_score(y_train, training_data_prediction)
print(f'R Squared Error (Training Data) - Lasso: {error_score}')

# Visualizing the actual prices vs predicted prices for training data (Lasso)
plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Data: Actual vs Predicted Prices (Lasso)')
#plt.show()

# Predicting selling prices on the test data using Lasso Regression
test_data_prediction = lass_reg_model.predict(x_test)

# Calculating the R-squared error for the test data prediction with Lasso
error_score = metrics.r2_score(y_test, test_data_prediction)
print(f'R Squared Error (Test Data) - Lasso: {error_score}')

# Visualizing the actual prices vs predicted prices for test data (Lasso)
plt.scatter(y_test, test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Test Data: Actual vs Predicted Prices (Lasso)')
plt.show()
