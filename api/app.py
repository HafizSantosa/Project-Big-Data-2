from flask import Flask, request, jsonify
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import os

# Initialize Flask app and Spark session
app = Flask(__name__)
spark = SparkSession.builder.appName("ModelPredictionApp").getOrCreate()

# Define the model path
MODEL_PATH = "/mnt/c/HaepjeT/Coolyeah/Big Data/Project 2 Kafka/ml-spark/models"  # Use absolute path if needed

# Function to load the model
def load_model(model_name):
    model_path = os.path.join(MODEL_PATH, model_name)
    
    print(f"Loading model from: {model_path}")  # Log path model
    
    # Check if the model exists at the given path
    if os.path.exists(model_path):
        try:
            # Try to load as a PipelineModel
            model = PipelineModel.load(model_path)
            print(f"Loaded PipelineModel: {model_name}")
            return model
        except Exception as e:
            print(f"Error loading pipeline model: {e}")
            try:
                # If not PipelineModel, try to load as RandomForest
                model = RandomForestClassificationModel.load(model_path)
                print(f"Loaded RandomForestClassificationModel: {model_name}")
                return model
            except Exception as e:
                print(f"Error loading RandomForest model: {e}")
                return None
    else:
        print(f"Model path does not exist: {model_path}")
        return None

def prepare_features(data):
    features = [
        data.get("shot_distance"),
        data.get("shot_clock"),
        data.get("shot_type_index"),
        data.get("player_name_index"),
        data.get("team_name_index")
    ]
    
    # Create DataFrame for the input data
    feature_vector = list(features)
    feature_df = spark.createDataFrame([tuple(feature_vector)], ["shot_distance", "shot_clock", "shot_type_index", "player_name_index", "team_name_index"])
    
    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=["shot_distance", "shot_clock", "shot_type_index", "player_name_index", "team_name_index"], outputCol="features")
    feature_df = assembler.transform(feature_df)
    
    return feature_df

@app.route('/predict-model/<model_id>', methods=['POST'])
def predict(model_id):
    data = request.json  # Get JSON data from request
    
    # Check for missing required fields
    if not all(key in data for key in ["shot_distance", "shot_clock", "shot_type_index", "player_name_index", "team_name_index"]):
        return jsonify({"error": "Missing required fields in input data."}), 400
    
    input_vector = prepare_features(data)
    
    # Load the model
    model_name = 'model_' + model_id
    model = load_model(model_name)
    
    if not model:
        return jsonify({"error": f"Model {model_id} not found"}), 404
    
    # Check if the model is a PipelineModel
    if isinstance(model, PipelineModel):
        classifier = model.stages[-1]
        if isinstance(classifier, RandomForestClassificationModel):
            prediction = classifier.transform(input_vector)
        else:
            return jsonify({"error": "Last stage is not a RandomForest classifier."}), 400
    elif isinstance(model, RandomForestClassificationModel):
        prediction = model.transform(input_vector)
    else:
        return jsonify({"error": "Unsupported model type."}), 400
    
    prediction_result = prediction.select("prediction").head()[0]  # Extract the prediction
    
    return jsonify({"prediction": prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
