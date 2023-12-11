Deep Learning Regression Model for Power Plant Efficiency Prediction
Overview
This repository contains a Python script (Stage-1.py) that implements a deep learning regression model using TensorFlow and Keras. The model is trained on the Combined Cycle Power Plant Dataset to predict power plant efficiency.

Prerequisites
Before running the script, ensure you have the following dependencies installed:

TensorFlow
NumPy
pandas
scikit-learn
matplotlib
You can install these dependencies using the following command:


pip install tensorflow numpy pandas scikit-learn matplotlib
Usage
Clone the repository:

git clone https://github.com/ToOLazYTOthinKOFAName0
/csrg.git
cd your-repository
Replace 'DL.csv' with the actual CSV file name containing your dataset.

Run the script:


python Stage-1.py
The script will train the model, evaluate its performance on the test set, and display the Mean Squared Error.
Configuration
You can update the hyperparameters in the script to experiment with different configurations for the neural network.


# Build the ANN model using TensorFlow and Keras
model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
Adjust the file name when saving the model if needed.

# Save the model if needed
# model.save("your_model.h5")
Visualizing Training History
The script includes code to visualize the training history using matplotlib. Uncomment the plotting code in the script to visualize the training and validation loss over epochs.
