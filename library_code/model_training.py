from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, output_units=1):
    """
    Builds and compiles an LSTM model.

    Parameters:
    - input_shape (tuple): The shape of the training dataset input (number of time steps, number of features).
    - lstm_units (int): Number of units in each LSTM layer, default is 50.
    - dropout_rate (float): Dropout rate for regularization, default is 0.2.
    - output_units (int): Number of units in the output layer, default is 1.

    Returns:
    - keras.Model: Compiled LSTM model.
    """

    model = Sequential()
    # First LSTM layer with Dropout regularisation
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers with Dropout regularisation
    for _ in range(2):
        model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(Dropout(dropout_rate))

    # Final LSTM layer
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=output_units))

    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, num_epochs=10, batch_size=32):
    """
    Trains the given LSTM model on the provided training data.

    Parameters:
    - model: The LSTM model to be trained.
    - X_train (np.ndarray): Training data features.
    - y_train (np.ndarray): Training data target variable.
    - num_epochs (int): Number of epochs for training, default is 50.
    - batch_size (int): Batch size for training, default is 32.

    Returns:
    - History: Object that records training history.
    """
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)
    return history