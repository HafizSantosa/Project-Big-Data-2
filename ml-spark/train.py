from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
import os
import time

# Path ke folder batch
batch_folder_path = '/mnt/c/HaepjeT/Coolyeah/Big Data/Project 2 Kafka/Batch'

# Membuat Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkML_NBA") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

# Fungsi untuk membaca data dari path
def read_data(path):
    try:
        return spark.read.csv(path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error reading data from {path}: {str(e)}")
        raise

def load_and_preprocess_data(batch_file_path):
    # Membaca dataset dari path
    df = read_data(batch_file_path)
    
    # Mengonversi kolom label (misalnya kolom 'EVENT_TYPE') menjadi format indeks numerik
    event_type_indexer = StringIndexer(inputCol="EVENT_TYPE", outputCol="label")
    indexed_model = event_type_indexer.fit(df)
    
    # Menampilkan kategori yang sudah terindeks
    print("Label untuk kolom EVENT_TYPE:")
    print(indexed_model.labels)
    
    df = indexed_model.transform(df)
    
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


def train_and_save_model(df, model_name):
    # Membagi data menjadi train dan test
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    # Model RandomForest dengan maxBins yang lebih besar
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxBins=170)
    
    # Membuat pipeline
    pipeline = Pipeline(stages=[rf])
    
    # Melatih model
    model = pipeline.fit(train_data)
    
    # Menyimpan model ke file lokal
    model_dir = f"file:///mnt/c/HaepjeT/Coolyeah/Big Data/Project 2 Kafka/ml-spark/models/{model_name}"
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
    try:
        batch_files = sorted(os.listdir(batch_folder_path))
    except Exception as e:
        print(f"Error listing files in {batch_folder_path}: {str(e)}")
        return
    
    batch_count = 0
    
    for batch_file in batch_files:
        batch_file_path = os.path.join(batch_folder_path, batch_file)
        
        # Memuat dan memproses data
        try:
            df = load_and_preprocess_data(batch_file_path)
        except Exception as e:
            print(f"Error processing file {batch_file}: {str(e)}")
            continue
        
        # Melatih dan menyimpan model
        model_name = f"model_{batch_count + 1}"
        try:
            train_and_save_model(df, model_name)
        except Exception as e:
            print(f"Error training and saving model {model_name}: {str(e)}")
            continue
        
        # Menampilkan status
        batch_count += 1
        print(f"Batch {batch_count} diproses dan model {model_name} telah dilatih")
        
        # Menunggu sejenak sebelum memproses batch berikutnya
        time.sleep(5)

# Menjalankan proses
process_batches()
spark.stop()
