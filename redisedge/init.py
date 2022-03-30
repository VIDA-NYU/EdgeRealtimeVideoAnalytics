import os# RedisEdge realtime video analytics initialization script
import sys
import time
import redis

sys.path.append('/app')
# os.chdir(os.path.dirname(__file__))
# from app 
import models


url = os.getenv('REDIS_URL') or 'redis://127.0.0.1:6379'
wins = [1, 5, 15]             # Downsampling windows
aggs = ['avg', 'min', 'max']  # Downsampling aggregates

metrics = ['read', 'preprocess', 'model', 'postprocess', 'total']

model_names = [
    'tinyyolov3',
]


def modelset(conn, url, model_name, gpu=False):
    path = os.path.join('model_weights', f'{model_name}.onnx')
    if url and not os.path.isfile(path):
        models.download_file(models.models[model_name.lower()].download_url, path)
    with open(path, 'rb') as f:
        res = conn.execute_command('AI.MODELSTORE', f'{model_name}:model', 'ONNX', ('GPU' if gpu is True else gpu) if gpu else 'CPU', f.read())
        print(res)

def register_camera(conn, camera_prefix='camera', camera_id=0, gpu=False):
    # Set up some vars
    input_key = f'{camera_prefix}:{camera_id}'  # Input video stream key name
    init_key = f'{input_key}:initialized'

    # # Check if this Redis instance had already been initialized
    # if conn.exists(init_key):
    #     print('Discovered evidence of a privious initialization - skipping.')
    #     return

    for n in model_names:
        modelset(conn, n, gpu=gpu)

    print('Creating timeseries keys and downsampling rules - ', end='')
    res = []                                                             # RedisTimeSeries replies list
    labels = ['LABELS', camera_prefix, camera_id, '__name__']  # A generic list of timeseries keys labels

    # Set up fps timeseries keys
    res.append(conn.execute_command(
        'TS.CREATE', F'{input_key}:in_fps', *labels, 'in_fps'))
    res.append(conn.execute_command(
        'TS.CREATE', F'{input_key}:out_fps', *labels, 'out_fps'))
    # res.append(conn.execute_command(
    #     'TS.CREATE', F'{input_key}:dropping_frame', *labels, 'dropping_frame'))
    # res.append(conn.execute_command(
    #     'TS.CREATE', F'{input_key}:ds_fps', *labels, 'ds_fps'))
    # res.append(conn.execute_command(
    #     'TS.CREATE', F'{input_key}:about_to_model', *labels, 'about_to_model'))
    # res.append(conn.execute_command(
    #     'TS.CREATE', F'{input_key}:finished_model', *labels, 'finished_model'))
    # Set up profiler timeseries keys
    for m in metrics:
        res.append(conn.execute_command(
             'TS.CREATE', F'{input_key}:prf_{m}', *labels, f'prf_{m}'))
    print(res)

    # Load the gear
    print('Loading gear - ')
    variations = [[input_key, m] for m in model_names]
    with open('/app/gear.py', 'r') as f:
        res = conn.execute_command('RG.PYEXECUTE', f.read() + f'''

variations = {variations!r}
for a in variations:
    main(*a)
        ''')
        print(res)

    # Lastly, set a key that indicates initialization has been performed
    print('Flag initialization as done - ', end='') 
    print(conn.set(init_key, 'most certainly.'))


def main():
    # pass
    for i in range(10):
        try:
            # Set up Redis connection
            conn = redis.from_url(url)
            if not conn.ping():
                raise redis.exceptions.ConnectionError('Redis unavailable')
        except redis.exceptions.ConnectionError as e:
            print('failed to connect to redis', type(e).__name__, str(e), file=sys.stderr)
            time.sleep(3)
            if i == 10 - 1:
                raise

    try:
        register_camera(conn)
    except redis.exceptions.ResponseError:
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    main()
