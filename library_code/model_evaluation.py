import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def model_testing(train_df, test_df, scaler, regressor):
    """
    Prepares the test dataset, scales it, and makes predictions using the trained model.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training data.
    test_df (pd.DataFrame): DataFrame containing the test data.
    scaler (MinMaxScaler): The scaler used for data normalization.
    regressor: The trained model for making predictions.

    Returns:
    np.ndarray: Predicted Nasdaq Index values.
    """

    # Combine training and testing data
    ds_total = pd.concat((train_df["Close"], test_df["Close"]), axis=0)

    # Prepare the inputs for prediction
    inputs = ds_total[len(ds_total) - len(test_df) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Making predictions
    predict_index = regressor.predict(X_test)
    predict_index = scaler.inverse_transform(predict_index)

    return predict_index

def plot_test_results(actual_index, predict_index):
    """
    Plots the actual and predicted Nasdaq Index values.

    Parameters:
    actual_index (np.ndarray): Array containing actual Nasdaq Index values.
    predict_index (np.ndarray): Array containing predicted Nasdaq Index values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual_index, color='red', label='Actual Nasdaq Index')
    plt.plot(predict_index, color='blue', label='Predicted Nasdaq Index')
    plt.title("Nasdaq Index Prediction")
    plt.xlabel('Time')
    plt.ylabel('Nasdaq Index')
    plt.legend()
    plt.show()

def model_evaluation(actual_index, predict_index):
    """
    Evaluates the model by comparing the trend accuracy of actual and predicted values.

    Parameters:
    actual_index (np.ndarray): Array containing actual Nasdaq Index values.
    predict_index (np.ndarray): Array containing predicted Nasdaq Index values.

    Returns:
    float: Percentage of time the predicted trend matches the actual trend.
    """

    actual = actual_index.flatten()
    predict = predict_index.flatten()
    actual = actual[:len(predict)]

    # Ensure both arrays have the same length
    if len(actual) != len(predict):
        raise ValueError("Actual and predicted arrays must be of the same length")

    df = pd.DataFrame({'actual': actual, 'predict': predict})

    # Calculate trend
    df['trend_actual'] = df['actual'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)
    df['trend_predict'] = df['predict'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)

    percentage_equal = (df['trend_actual'] == df['trend_predict']).mean() * 100
    return percentage_equal
