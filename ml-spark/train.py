from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
import os
import time

# Path ke folder batch
batch_folder_path = "C:\\HaepjeT\\Coolyeah\\Big Data\\Project 2 Kafka\\Batch"

# Membuat Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkML_NBA") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

# Fungsi untuk membaca data dari path
def read_data(path):
    return spark.read.csv(path, header=True, inferSchema=True)

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(batch_file_path):
    # Membaca dataset dari path
    df = read_data(batch_file_path)
    
    # Mengonversi kolom label (misalnya kolom 'EVENT_TYPE') menjadi format indeks numerik
    label_indexer = StringIndexer(inputCol="EVENT_TYPE", outputCol="label").fit(df)
    df = label_indexer.transform(df)
    
    # Mengonversi kolom kategori lain yang diperlukan
    categorical_columns = ['SHOT_TYPE', 'PLAYER_NAME', 'TEAM_NAME']
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(df) for col in categorical_columns]
    
    for indexer in indexers:
        df = indexer.transform(df)
    
    # Menghapus kolom kategori asli
    df = df.drop(*categorical_columns)
    
    # Menggunakan kolom fitur numerik yang telah ditentukan
    feature_columns = ['SHOT_DISTANCE', 'SHOT_TYPE_index', 'PLAYER_NAME_index', 'TEAM_NAME_index']
    
    # Merangkai kolom fitur menjadi vektor fitur tunggal
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    
    # Memilih hanya kolom 'features' dan 'label' untuk pelatihan
    df = df.select("features", "label")
    
    return df

# Fungsi untuk melatih dan menyimpan model
def train_and_save_model(df, model_name):
    # Membagi data menjadi train dan test
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Model RandomForest dengan maxBins yang lebih besar
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxBins=150)
    
    # Membuat pipeline
    pipeline = Pipeline(stages=[rf])
    
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

# Fungsi untuk memproses semua file batch dalam folder batch
def process_batches():
    batch_files = sorted(os.listdir(batch_folder_path))
    batch_count = 0
    
    for batch_file in batch_files:
        batch_file_path = os.path.join(batch_folder_path, batch_file)
        
        # Memuat dan memproses data
        df = load_and_preprocess_data(batch_file_path)
        
        # Melatih dan menyimpan model
        model_name = f"model_{batch_count + 1}"
        train_and_save_model(df, model_name)
        
        # Menampilkan status
        batch_count += 1
        print(f"Batch {batch_count} diproses dan model {model_name} telah dilatih")
        
        # Menunggu sejenak sebelum memproses batch berikutnya
        time.sleep(5)

# Menjalankan proses
process_batches()
spark.stop()
