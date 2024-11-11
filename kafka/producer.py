from time import sleep
from json import dumps
from kafka import KafkaProducer
import csv
import random

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: dumps(x).encode('utf-8')
)

dataset = '../NBA_2024_Shots.csv'

with open(dataset, 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        producer.send('kafka-server', value=row)
        print(row)
        sleep(random.uniform(0.05, 0.5))

# producer.flush()
# producer.close()