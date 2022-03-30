import os
import io
import json
import time
import numpy as np
from PIL import Image, ImageDraw

import redis
import models

red = redis.from_url(os.getenv('REDIS_URL') or 'redis://127.0.0.1:6379')
print('ping', red.ping())


def main(device_name='camera:0', model_name='tinyyolov2', field='boxes', maxlen=1000):
    model = models.get_model(model_name)

    previous = None
    last_duplicate = False
    while True:
        cmsg = red.xrevrange(device_name, count=1)
        if not cmsg:
            time.sleep(1)
            continue
        id, data = cmsg[0]
        if previous == id:
            if not last_duplicate:
                print('Duplicate. skipping...', id, time.time())
            time.sleep(1)
            last_duplicate = True
            continue
        previous = id
        last_duplicate = False

        print(id, len(data['image'.encode('utf-8')]), flush=True)
        # time.sleep(1)

        img = np.array(Image.open(io.BytesIO(cmsg[0][1]['image'.encode('utf-8')])))
        boxes = model(img)[0]
        print(boxes)
        try:
            _id = red.xadd(f'{device_name}:{model_name}', { field: json.dumps(boxes) }, id, maxlen=maxlen)
        except redis.exceptions.ResponseError:
            import traceback
            traceback.print_exc()
            time.sleep(1)


if __name__ == '__main__':
    import fire
    fire.Fire(main)