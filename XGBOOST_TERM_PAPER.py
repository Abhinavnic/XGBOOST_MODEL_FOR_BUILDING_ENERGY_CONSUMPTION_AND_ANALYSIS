import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from google.colab import files

upload=files.upload()

da = pd.read_excel("dataset for buildings energy consumption of 3840 records (1).xlsx")
da

X = da.iloc[:, list(range(0, 10)) + list(range(15, 23))]
X

Y = da.iloc[:, 13]
Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2529)

dtrain = xgb.DMatrix(X_train.values, label=Y_train.values)
dtest = xgb.DMatrix(X_test, label=Y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

bst = xgb.train(params, dtrain, num_boost_round=100)

Y_pred = bst.predict(dtest)

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

threshold = Y_test.mean()
Y_test_binary = (Y_test > threshold).astype(int)
Y_pred_binary = (Y_pred > threshold).astype(int)
accuracy = accuracy_score(Y_test_binary, Y_pred_binary)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")

