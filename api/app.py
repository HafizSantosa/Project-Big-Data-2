from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import os

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FlaskSparkMLApp_NBA") \
    .master("local[*]") \
    .getOrCreate()

MODEL_PATH = "ml-spark/models/"  # Modify this path if necessary

# Function to load the model
def load_model(model_name):
    model_path = os.path.join(MODEL_PATH, model_name)
    if os.path.exists(model_path):
        return RandomForestClassificationModel.load(model_path)
    else:
        return None

# Function to prepare feature vector based on NBA dataset columns
def prepare_features(data):
    features = [
        data.get("shot_distance", 0),
        data.get("shot_clock", 0),
        data.get("shot_type_index", 0),
        data.get("player_name_index", 0),
        data.get("team_name_index", 0)
    ]
    return Vectors.dense(features)

# Endpoint for prediction
@app.route("/predict-model/<model_id>", methods=["POST"])
def predict(model_id):
    data = request.json  # Get JSON data from request
    input_vector = prepare_features(data)  # Prepare features

    model_name = 'model_' + model_id
    model = load_model(model_name)  # Load model

    # If model is not found, return an error
    if not model:
        return jsonify({"error": f"Model {model_id} tidak ditemukan"}), 404

    # Make prediction
    prediction = model.predict(input_vector)

    return jsonify({"model": int(model_id), "shot_result": int(prediction)})

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)

    spark.stop()
