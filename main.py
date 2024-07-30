import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('kc_house_data.csv')

X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, np.log1p(y_train))  # Apply log transformation to y_train

y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)  # Back-transform predictions

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

def predict_price(model, scaler):
    bedrooms = float(input("Enter number of bedrooms: "))
    bathrooms = float(input("Enter number of bathrooms: "))
    sqft_living = float(input("Enter square footage of living space: "))
    sqft_lot = float(input("Enter square footage of lot space: "))
    floors = float(input("Enter number of floors: "))
    
    user_input = np.array([bedrooms, bathrooms, sqft_living, sqft_lot, floors])
    user_input_df = pd.DataFrame([user_input], columns=X.columns)
    
    user_input_scaled = scaler.transform(user_input_df)
    
    predicted_price_log = model.predict(user_input_scaled)
    predicted_price = np.expm1(predicted_price_log)  # Back-transform prediction
    
    print(f'The predicted price for the house is: ${predicted_price[0]:.2f}')

predict_price(model, scaler)
