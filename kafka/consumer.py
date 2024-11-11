from kafka import KafkaConsumer
from json import loads

consumer = KafkaConsumer(
    'kafka-server',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

message_buffer = []
buffer_size = 15000 # Jumlah data per batch, bisa disesuaikan
batch = 0
counter = 0

header = 'SEASON_1,SEASON_2,TEAM_ID,TEAM_NAME,PLAYER_ID,PLAYER_NAME,POSITION_GROUP,POSITION,GAME_DATE,GAME_ID,HOME_TEAM,AWAY_TEAM,EVENT_TYPE,SHOT_MADE,ACTION_TYPE,SHOT_TYPE,BASIC_ZONE,ZONE_NAME,ZONE_ABB,ZONE_RANGE,LOC_X,LOC_Y,SHOT_DISTANCE,QUARTER,MINS_LEFT,SECS_LEFT\n'

for message in consumer:
    if counter == 0:
        output = open(f'../Batch/batch{batch}.csv', 'w', encoding='utf-8')
        output.write(header)

    data = message.value
    row = ','.join(str(data.get(col, '')) for col in header.strip().split(',')) + '\n'
    output.write(row)
    print(data)
    counter += 1

    if counter >= buffer_size:
        output.close()
        counter = 0
        batch += 1