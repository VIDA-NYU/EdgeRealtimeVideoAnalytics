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


def main(device_name='camera:0', model_name='yolov3', field='boxes', maxlen=1000):
    model = models.get_model(model_name)

    previous = None
    last_duplicate = 0
    while True:
        cmsg = red.xrevrange(device_name, count=1)
        if not cmsg:
            time.sleep(0.1)
            continue
        id, data = cmsg[0]
        if previous == id:
            if not last_duplicate:
                print('Duplicate. skipping...', id, time.time())
            time.sleep(1 if last_duplicate > 30 else 0.1)
            last_duplicate += 1
            continue
        previous = id
        last_duplicate = 0

        # print(id, len(data['image'.encode('utf-8')]), flush=True)
        # time.sleep(1)

        img = np.array(Image.open(io.BytesIO(cmsg[0][1]['image'.encode('utf-8')])))
        boxes = model(img)[0]
        print(id, len(data['image'.encode('utf-8')]), boxes)
        try:
            _id = red.xadd(
                f'{device_name}:boxes', 
                { field: json.dumps(boxes, cls=NJSONEncoder) }, 
                id, maxlen=maxlen)  # f'{device_name}:{model_name}'
        except redis.exceptions.ResponseError:
            import traceback
            traceback.print_exc()
            time.sleep(1)


class NJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


if __name__ == '__main__':
    import fire
    fire.Fire(main)