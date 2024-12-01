The code sets up a Flask web application for anomaly detection using a Convolutional Neural Network (CNN). It begins by importing necessary libraries such as numpy, pandas, tensorflow, flask, and others for data manipulation, machine learning, and web application creation. The Flask app is initialized, and ngrok is used to enable remote access.

A function generates synthetic data to simulate patient and anomaly datasets. The data is then preprocessed by scaling it with StandardScaler and splitting it into training and testing sets. The CNN model is created with Conv1D layers, max-pooling layers, and dense layers, followed by compiling it for training with the Adam optimizer and binary cross-entropy loss.

The model is trained on the preprocessed data using the fit() method, which runs for 10 epochs with a batch size of 32. The Flask app defines two routes: the root (/) serves a simple HTML page with a button that triggers anomaly detection, while the /predict route handles GET requests for predictions. In this route, 100 random samples from the test set are selected, and the model predicts whether each sample is "Normal" or "Anomaly" along with a confidence score.

Finally, the app is run, making it available for users to interact with the anomaly detection model via the web interface.


Explanation:
Libraries Imported: The code starts by importing necessary libraries such as numpy, tensorflow, flask, etc.
Flask App Setup: The Flask app is initialized and the ngrok is used for remote access during development.
Generate Synthetic Data: Synthetic data for normal patients and anomalies is generated.
Preprocess Data: The data is scaled using StandardScaler, and then it is split into training and testing datasets.
Create CNN Model: A Convolutional Neural Network (CNN) model is created using Conv1D, MaxPooling1D, and Dense layers.
Train the Model: The CNN model is trained on the preprocessed data, using fit().
Flask Routes:
The / route serves an HTML page with a button for anomaly detection.
The /predict route handles prediction requests and returns results for randomly selected samples.
Prediction Logic: For the /predict route, the model predicts whether a sample is "Normal" or "Anomaly" based on the test data.
Run Flask: Finally, the Flask app is run to make the web application accessible.
This diagram illustrates the flow of data from generating synthetic datasets to serving predictions through a web interface.

+------------------+
|  Import Libraries|
+------------------+
        |
        V
        
+------------------+
| Flask App Setup  |
+------------------+
        |
        V
        
+---------------------------+
| Generate Synthetic Data   |
+---------------------------+
        |
        V
        
+----------------------------+
| Preprocess Data            |
| - Scale Data               |
| - Split Data               |
+----------------------------+
        |
        V
+----------------------------+
| Create CNN Model           |
| - Conv1D Layers            |
| - MaxPooling1D Layers      |
| - Dense Layers             |
| - Compile Model            |
+----------------------------+
        |
        V
        
+----------------------------+
| Train CNN Model            |
| - Fit on Train Data        |
| - Validate on Test Data    |
+----------------------------+
        |
        V
        
+----------------------------+
| Flask Routes               |
| - Route '/' for UI         |
| - Route '/predict' for     |
|   Anomaly Detection        |
+----------------------------+
        |
        V
+----------------------------+
| Prediction Function        |
| - Select Random Samples    |
| - Predict Using Trained    |
|   Model                    |
| - Return Results (Anomaly/ |
|   Normal) with Confidence  |
+----------------------------+
        |
        V
        
+----------------------------+
| Run Flask App              |
| - Host Web Application     |
+----------------------------+


import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify
from flask_ngrok import run_with_ngrok
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Function to generate synthetic datasets
def generate_data():
    patient_data = np.random.randn(800, 20)
    patient_labels = np.zeros(800)  # Normal data labeled as 0
    anomaly_data = np.random.randn(200, 20) + 5  # Shifted data to simulate anomalies
    anomaly_labels = np.ones(200)  # Anomalies labeled as 1
    return {'data': patient_data, 'labels': patient_labels}, {'data': anomaly_data, 'labels': anomaly_labels}

# Preprocess data: scale and split into training/testing sets
def preprocess_data(data, labels):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Define a CNN model for anomaly detection
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Reshape((input_shape[0], 1)),  # For Conv1D processing
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2, padding='same'),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate datasets
patient_data, anomaly_data = generate_data()
data = np.vstack((patient_data['data'], anomaly_data['data']))
labels = np.hstack((patient_data['labels'], anomaly_data['labels']))

# Preprocess the combined dataset
X_train, X_test, y_train, y_test, scaler = preprocess_data(data, labels)

# Create and train the CNN model
model = create_cnn_model((X_train.shape[1],))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Flask routes
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Anomaly Detection</title>
    </head>
    <body>
        <h1>Anomaly Detection</h1>
        <p>Click the button to detect 100 anomalies in predefined data.</p>
        <button onclick="detectAnomalies()">Detect 100 Anomalies</button>
        <div id="results"></div>

        <script>
            async function detectAnomalies() {
                const response = await fetch('/predict', {
                    method: 'GET',
                });
                const data = await response.json();
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = ''; // Clear previous results
                data.results.forEach((result, index) => {
                    const resultElement = document.createElement('p');
                    resultElement.innerHTML = `Sample ${index + 1}: Result: ${result.result}, Confidence: ${result.confidence}`;
                    resultsDiv.appendChild(resultElement);
                });
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['GET'])
def predict():
    try:
        results = []
        # Select 100 random samples from the test set
        for _ in range(100):
            sample_index = np.random.randint(0, len(X_test))  # Select random index from test data
            input_data = X_test[sample_index].reshape(1, -1)  # Reshape to (1, 20)
            
            # Apply scaler
            input_array = scaler.transform(input_data)

            # Predict using the trained model
            prediction = model.predict(input_array)
            result = "Anomaly" if prediction[0][0] > 0.5 else "Normal"
            confidence = float(prediction[0][0])

            results.append({'result': result, 'confidence': confidence})

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run()


Epoch 1/10
/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/input_layer.py:26: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
  warnings.warn(
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 10ms/step - accuracy: 0.9768 - loss: 0.3704 - val_accuracy: 1.0000 - val_loss: 0.0044
Epoch 2/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 1.0000 - loss: 0.0019 - val_accuracy: 1.0000 - val_loss: 1.8976e-04
Epoch 3/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 1.0000 - loss: 1.6857e-04 - val_accuracy: 1.0000 - val_loss: 1.1276e-04
Epoch 4/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 1.0000 - loss: 1.1655e-04 - val_accuracy: 1.0000 - val_loss: 9.5260e-05
Epoch 5/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 1.0000 - loss: 9.8937e-05 - val_accuracy: 1.0000 - val_loss: 8.3341e-05
Epoch 6/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 1.0000 - loss: 8.7370e-05 - val_accuracy: 1.0000 - val_loss: 7.3126e-05
Epoch 7/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 1.0000 - loss: 7.6680e-05 - val_accuracy: 1.0000 - val_loss: 6.4406e-05
Epoch 8/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 1.0000 - loss: 6.3935e-05 - val_accuracy: 1.0000 - val_loss: 5.6953e-05
Epoch 9/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 1.0000 - loss: 6.4862e-05 - val_accuracy: 1.0000 - val_loss: 5.0346e-05
Epoch 10/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 1.0000 - loss: 5.2397e-05 - val_accuracy: 1.0000 - val_loss: 4.4927e-05
 * Serving Flask app '__main__'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
INFO:werkzeug:Press CTRL+C to quit
 * Running on http://f802-34-23-85-163.ngrok-free.app
 * Traffic stats available on http://127.0.0.1:4040
INFO:werkzeug:127.0.0.1 - - [01/Dec/2024 17:33:58] "GET / HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [01/Dec/2024 17:33:59] "GET /favicon.ico HTTP/1.1" 404 -
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
INFO:werkzeug:127.0.0.1 - - [01/Dec/2024 17:34:07] "GET /predict HTTP/1.1" 200 -

Results

Anomaly Detection
Click the button to detect 100 anomalies in predefined data.

Detect 100 Anomalies
Sample 1: Result: Normal, Confidence: 0.000049769463657867163

Sample 2: Result: Normal, Confidence: 0.00005842103928443976

Sample 3: Result: Anomaly, Confidence: 0.7579156756401062

Sample 4: Result: Normal, Confidence: 0.00001834291106206365

Sample 5: Result: Anomaly, Confidence: 0.8342196941375732

Sample 6: Result: Normal, Confidence: 0.00006380449485732242

Sample 7: Result: Normal, Confidence: 0.00002406512612651568

Sample 8: Result: Anomaly, Confidence: 0.7476329207420349

Sample 9: Result: Anomaly, Confidence: 0.7373083829879761

Sample 10: Result: Normal, Confidence: 0.00005300295015331358

Sample 11: Result: Normal, Confidence: 0.00003806871245615184

Sample 12: Result: Normal, Confidence: 0.00003606659811339341

Sample 13: Result: Normal, Confidence: 0.00002688214954105206

Sample 14: Result: Normal, Confidence: 0.000016343161405529827

Sample 15: Result: Normal, Confidence: 0.00001299632822338026

Sample 16: Result: Normal, Confidence: 0.00004053608427057043

Sample 17: Result: Normal, Confidence: 0.00001577386683493387

Sample 18: Result: Anomaly, Confidence: 0.7543355226516724

Sample 19: Result: Anomaly, Confidence: 0.7838546633720398

Sample 20: Result: Normal, Confidence: 0.000025609728254494257

Sample 21: Result: Anomaly, Confidence: 0.6813217997550964

Sample 22: Result: Normal, Confidence: 0.000029373884899541736

Sample 23: Result: Normal, Confidence: 0.000025595712941139936

Sample 24: Result: Normal, Confidence: 0.00007870927947806194

Sample 25: Result: Anomaly, Confidence: 0.6381637454032898

Sample 26: Result: Normal, Confidence: 0.00002464952558511868

Sample 27: Result: Anomaly, Confidence: 0.673441469669342

Sample 28: Result: Normal, Confidence: 0.000049769463657867163

Sample 29: Result: Normal, Confidence: 0.00002383024366281461

Sample 30: Result: Anomaly, Confidence: 0.5534269213676453

Sample 31: Result: Normal, Confidence: 0.000025595712941139936

Sample 32: Result: Anomaly, Confidence: 0.7316622734069824

Sample 33: Result: Normal, Confidence: 0.00003261708116042428

Sample 34: Result: Normal, Confidence: 0.000029373884899541736

Sample 35: Result: Anomaly, Confidence: 0.7478422522544861

Sample 36: Result: Normal, Confidence: 0.000018908225683844648

Sample 37: Result: Normal, Confidence: 0.00003034205292351544

Sample 38: Result: Anomaly, Confidence: 0.6167935132980347

Sample 39: Result: Normal, Confidence: 0.00008919355605030432

Sample 40: Result: Normal, Confidence: 0.000025595712941139936

Sample 41: Result: Normal, Confidence: 0.000024623257559142075

Sample 42: Result: Anomaly, Confidence: 0.7478422522544861

Sample 43: Result: Normal, Confidence: 0.00009588446846464649

Sample 44: Result: Normal, Confidence: 0.00006299706728896126

Sample 45: Result: Normal, Confidence: 0.00004847462696488947

Sample 46: Result: Normal, Confidence: 0.00004173493289272301

Sample 47: Result: Normal, Confidence: 0.00003170664786011912

Sample 48: Result: Normal, Confidence: 0.00003861044388031587

Sample 49: Result: Normal, Confidence: 0.4765843152999878

Sample 50: Result: Normal, Confidence: 0.00003170664786011912

Sample 51: Result: Normal, Confidence: 0.00005119629713590257

Sample 52: Result: Normal, Confidence: 0.00008919355605030432

Sample 53: Result: Normal, Confidence: 0.00005243836130830459

Sample 54: Result: Normal, Confidence: 0.000030613577109761536

Sample 55: Result: Normal, Confidence: 0.000050538914365461096

Sample 56: Result: Normal, Confidence: 0.0000378003969672136

Sample 57: Result: Normal, Confidence: 0.00003396101965336129

Sample 58: Result: Anomaly, Confidence: 0.7478422522544861

Sample 59: Result: Normal, Confidence: 0.000029905748306191526

Sample 60: Result: Normal, Confidence: 0.000027591260732151568

Sample 61: Result: Normal, Confidence: 0.00011552374780876562

Sample 62: Result: Normal, Confidence: 0.000031348445190815255

Sample 63: Result: Normal, Confidence: 0.00004010260454379022

Sample 64: Result: Normal, Confidence: 0.000029704115149797872

Sample 65: Result: Normal, Confidence: 0.0000591918287682347

Sample 66: Result: Normal, Confidence: 0.00008470976172247902

Sample 67: Result: Anomaly, Confidence: 0.5643036365509033

Sample 68: Result: Normal, Confidence: 0.00003171989374095574

Sample 69: Result: Normal, Confidence: 0.00005973592124064453

Sample 70: Result: Normal, Confidence: 0.4914456605911255

Sample 71: Result: Normal, Confidence: 0.00001841006633185316

Sample 72: Result: Normal, Confidence: 0.00005243836130830459

Sample 73: Result: Normal, Confidence: 0.00004507966514211148

Sample 74: Result: Normal, Confidence: 0.00008637534483568743

Sample 75: Result: Normal, Confidence: 0.00003559271863196045

Sample 76: Result: Normal, Confidence: 0.000027591260732151568

Sample 77: Result: Normal, Confidence: 0.000025595712941139936

Sample 78: Result: Normal, Confidence: 0.000016812486137496307

Sample 79: Result: Normal, Confidence: 0.0000591918287682347

Sample 80: Result: Normal, Confidence: 0.00007846407970646396

Sample 81: Result: Normal, Confidence: 0.4765843152999878

Sample 82: Result: Anomaly, Confidence: 0.7084518671035767

Sample 83: Result: Anomaly, Confidence: 0.7395805716514587

Sample 84: Result: Normal, Confidence: 0.00010933860903605819

Sample 85: Result: Normal, Confidence: 0.00001841006633185316

Sample 86: Result: Normal, Confidence: 0.00002305674206581898

Sample 87: Result: Normal, Confidence: 0.00010767804633360356

Sample 88: Result: Normal, Confidence: 0.00004075259494129568

Sample 89: Result: Normal, Confidence: 0.00005831384260090999

Sample 90: Result: Normal, Confidence: 0.00006919504085090011

Sample 91: Result: Normal, Confidence: 0.00004507966514211148

Sample 92: Result: Anomaly, Confidence: 0.7838546633720398

Sample 93: Result: Normal, Confidence: 0.00006390314229065552

Sample 94: Result: Normal, Confidence: 0.00005122290531289764

Sample 95: Result: Normal, Confidence: 0.00006230593135114759

Sample 96: Result: Normal, Confidence: 0.00004871016790275462

Sample 97: Result: Normal, Confidence: 0.00006919504085090011

Sample 98: Result: Normal, Confidence: 0.00005842103928443976

Sample 99: Result: Normal, Confidence: 0.00006919504085090011

Sample 100: Result: Anomaly, Confidence: 0.824809730052948

