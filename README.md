# Stores_Sales_Prediction

## Project Overview
The Stores Sales Prediction project aims to develop a machine learning solution that can accurately forecast future sales for different stores and products. By leveraging historical sales data, store characteristics and product information, the system will provide valuable insights to help retailers optimize inventory management, adjust marketing strategies and improve overall business performance.

## Features
- Data Preprocessing: Comprehensive data cleaning, handling of missing values, and feature engineering to prepare the dataset for model training.
-Machine Learning Model: Implementation of a Gradient Boosting Regressor model to predict store item sales.
- Streamlit Application: User-friendly web interface for interactive predictions.
- Model Deployment: Trained models are serialized for integration into the Streamlit app, ensuring scalability.

## Technical Details
- **Programming Language:** Python
- **Libraries:** Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn
Web Application: Streamlit
- **Model:** Gradient Boosting Regressor
- **Deployment:** Streamlit web application

## Project Structure
stores-sales-prediction/

├── data/

│   ├── train.csv

│   └── test.csv

│   └──train_data_edited.csv

├── apps.py

├── model.pkl

├── README.md

├── requirements.txt

├── sales_train_main.ipynb



# Code Workflow

1. Data Preprocessing
Numeric columns are scaled as needed.
Categorical columns (e.g., Outlet Identifier) are one-hot encoded.

2. Model Loading
A pre-trained machine learning model is loaded.

3. Sales Prediction

The model predicts sales based on preprocessed user inputs.
Log-transformed predictions are reversed to actual sales values using exponential transformation.

# Model Performance
Model	            MAE	     MSE    R² Score

Linear Regression	0.4157	0.2843	0.7305

Gradient Boosting	0.3884	0.2533	0.7536



# Input Features
*Given* - 
1. Item_Identifier: Unique product ID
2. Item_Weight: Weight of the product
3. Item_Fat_Content: Indicates whether the product is low fat or not
4. Item_Visibility: Percentage of total display area allocated to the product in the store
5. Item_Type: Category to which the product belongs
6. Item_MRP: Maximum Retail Price (list price) of the product
7. Outlet_Identifier: Unique store ID
8. Outlet_Establishment_Year: The year the store was established
9. Outlet_Size: Size of the store in terms of ground area covered
10. Outlet_Location_Type: Type of city in which the store is located
11. Outlet_Type: Type of outlet (e.g., grocery store or supermarket)

*Input needed for the model after preproceesing* - 
1. Item_Visibility,
2. Item_MRP,
3. Outlet_Identifier,
4. Outlet_Size,
5. Outlet_Location_Type,
6. Outlet_Type,

*output* - Item_Outlet_Sales: Sales of the product in the particular store (target variable)