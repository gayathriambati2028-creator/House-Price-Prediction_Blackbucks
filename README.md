
This project is a House Price Prediction System developed using Python and Machine Learning.
 It uses a dataset named house_price_dataset.csv.
 The dataset contains information such as:
  * City
  * Area size
  * Number of rooms
  * Furnishing type
  * Crime rate
  * Market price
  The main goal is to predict the price of a house based on its features.
  The data is cleaned before training the model.
  Text data such as City,Locality and Furnishing are converted into numerical values using one-hot encoding(pd.get_dummies).
  One-Hot Encoding is used to convert categorical (text) data into numerical (0 and 1) values so that machine learning algorithms can understand and use the data.
  The dataset is divided into 80% training data and 20% testing data.
  A Linear Regression model from Scikit-learn is used to train the data.
  The trained model predicts house prices using the testing data.
  The model performance is evaluated using:
Mean Absolute Error (MAE) – measures the average prediction error.
R² Score – measures how well the model predicts house prices.
