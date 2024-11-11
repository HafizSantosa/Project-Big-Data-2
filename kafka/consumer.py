from kafka import KafkaConsumer
from json import loads
import os

consumer = KafkaConsumer(
    'kafka-server',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

# Tentukan jalur direktori batch
batch_folder_path = 'C:\\HaepjeT\\Coolyeah\\Big Data\\Project 2 Kafka\\Batch'
if not os.path.exists(batch_folder_path):
    os.makedirs(batch_folder_path)

message_buffer = []
buffer_size = 50000  # Jumlah data per batch
batch = 0
counter = 0

header = 'SEASON_1,SEASON_2,TEAM_ID,TEAM_NAME,PLAYER_ID,PLAYER_NAME,POSITION_GROUP,POSITION,GAME_DATE,GAME_ID,HOME_TEAM,AWAY_TEAM,EVENT_TYPE,SHOT_MADE,ACTION_TYPE,SHOT_TYPE,BASIC_ZONE,ZONE_NAME,ZONE_ABB,ZONE_RANGE,LOC_X,LOC_Y,SHOT_DISTANCE,QUARTER,MINS_LEFT,SECS_LEFT\n'

# Pembukaan file batch
output = open(os.path.join(batch_folder_path, f'batch{batch}.csv'), 'w', encoding='utf-8')
output.write(header)

try:
    for message in consumer:
        data = message.value
        row = ','.join(str(data.get(col, '')) for col in header.strip().split(',')) + '\n'
        output.write(row)
        print(data)
        counter += 1

        if counter >= buffer_size:
            output.close()
            counter = 0
            batch += 1
            # Buat file baru untuk batch berikutnya
            output = open(os.path.join(batch_folder_path, f'batch{batch}.csv'), 'w', encoding='utf-8')
            output.write(header)

except ValueError as e:
    print(f"Error: {e}")
finally:
    output.close()
