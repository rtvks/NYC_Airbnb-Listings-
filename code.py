import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from urllib.request import urlretrieve
urlretrieve('http://data.insideairbnb.com/united-states/ny/new-york-city/2023-03-06/visualisations/listings.csv','NYC_AIRBNB.csv')

data = pd.read_csv('NYC_AIRBNB.csv',low_memory = False)

# Data Cleaning
data.dropna(inplace=True)  # Remove rows with missing values

#plotting 
sns.set(style="ticks", color_codes=True)
sns.pairplot(nyc_df_clean, x_vars=["latitude", "longitude", "price"], y_vars=["latitude", "longitude", "price"])
plt.show()

# Exploratory Data Analysis
average_prices_by_location = data.groupby('neighbourhood')['price'].mean()
property_type_counts = data['property_type'].value_counts()

# Feature Engineering (Example: Creating dummy variables)
data = pd.get_dummies(data, columns=['property_type', 'room_type'])

# Splitting data into training and testing sets
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Visualization (Example: Scatter plot of predicted vs. actual prices)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()

# Example of predicting a price for a new listing
new_listing = pd.DataFrame({
    'neighbourhood': ['Upper West Side'],
    'availability_365': [100],
    'property_type_Apartment': [1],
    'property_type_House': [0],
    'room_type_Entire home/apt': [1],
    'room_type_Private room': [0]
})

predicted_price = model.predict(new_listing)
print("Predicted Price:", predicted_price[0])
