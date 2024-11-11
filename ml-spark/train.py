from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Membuat SparkSession dengan konfigurasi yang tepat
spark = SparkSession.builder \
    .appName("KafkaSparkML_NBA") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

# Membaca data
def read_data(path):
    return spark.read.csv(path, header=True, inferSchema=True)

# Mempersiapkan dan melatih model
def train_and_save_model(df, model_name):
    # Menyiapkan data untuk pelatihan
    feature_columns = df.columns[:-1]  # Semua kolom kecuali label
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    
    # Model RandomForest
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    
    # Membuat pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Membagi data menjadi train dan test
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Melatih model
    model = pipeline.fit(train_data)
    
    # Menyimpan model ke file lokal
    model_dir = f"file:///c:/HaepjeT/Coolyeah/Big Data/Project 2 Kafka/ml-spark/models/{model_name}"
    model.write().overwrite().save(model_dir)
    print(f"Model disimpan di {model_dir}")
    
    # Evaluasi model
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    
    accuracy = evaluator.evaluate(predictions)
    print(f"Akurasi Model: {accuracy}")
    
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = f1_evaluator.evaluate(predictions)
    print(f"F1 Score: {f1_score}")

# Memproses batch dan melatih model
def process_batches():
    data_path = "file:///c:/HaepjeT/Coolyeah/Big Data/Project 2 Kafka/data/nba_data.csv"
    df = read_data(data_path)
    model_name = "model_1"
    train_and_save_model(df, model_name)

# Menjalankan proses
process_batches()
