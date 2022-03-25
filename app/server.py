# RedisEdge realtime video analytics web server
import os
import io
import time
import asyncio
# from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw

# import redis
from redis import asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, Response, PlainTextResponse
from fastapi.templating import Jinja2Templates

# Set up Redis connection
# url = urlparse(os.getenv('REDIS_URL') or 'redis://127.0.0.1:6379')
# red = redis.Redis(host=url.hostname, port=url.port)
red = aioredis.from_url(os.getenv('REDIS_URL') or 'redis://127.0.0.1:6379')


app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open('data/technical_difficulties_please_stand_by_television_test_pattern.jpg', 'rb') as f:
    technical_difficulties = f.read()

_, blank = cv2.imencode('.jpg', np.zeros((600, 400, 3)))
blank = blank.tobytes()


RUNNING = True

@app.on_event("startup")
async def startup_event():
    if not await red.ping():
        raise Exception('Redis unavailable')



@app.get('/')
def index():
    return 'Hi!'



@app.get('/redis')
def redis_info():
    '''Debug info about redis'''
    try:
        pool = red.connection_pool
        inuse = '\n'.join(map(repr, list(pool._in_use_connections)[:30]))
        avail = '\n'.join(map(repr, list(pool._available_connections)[:30]))
        return PlainTextResponse(f'''
Max Connections: {pool.max_connections}
# Created Connections: {pool._created_connections}
# Available: {len(pool._available_connections)}
# In Use: {len(pool._in_use_connections)}

In Use:
{inuse}

Available:
{avail}
        ''')
    except AttributeError:
        import traceback
        return PlainTextResponse(traceback.format_exc())
# Locked: {pool._lock.locked}








@app.get('/webcam')
def webcam(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})



@app.websocket('/video/in')
async def video_input(websocket: WebSocket, name='camera:0', field='image', maxlen=10000):
    await websocket.accept()
    try:
        count = 0
        while True:
            data = await websocket.receive_bytes()
            # print(len(data))
            _id = await red.xadd(name, { 'count': count, field: data }, maxlen=maxlen)
            # _id = await (
            #     red.pipeline()
            #        .xadd(name, { 'count': count, field: data }, maxlen=maxlen)
            #        .execute())
    except WebSocketDisconnect:
        print('Video Input Disconnected')



@app.get('/video')
def video_feed(name='camera:0', boxes='camera:0:yolo', field='image'):
    print('creating response')
    async def stream():
        i = 0
        print('started response')
        s = RedisImageStream(name, boxes, field)
        while True:
            # t0 = time.time()
            i += 1
            print('awaiting response')
            f = await s.latest()
            print(1111, i, 'none' if f is None else len(f))
            if not RUNNING or f is None:
                yield img_replace_frame(technical_difficulties)
                continue
            if not i % 100: print(i)
            await asyncio.sleep(0.04)
            yield img_replace_frame(f)
        print('ending response')

    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')


def img_replace_frame(frame):
    return (
        b'--frame\r\n'
        b'Pragma-directive: no-cache\r\n'
        b'Cache-directive: no-cache\r\n'
        b'Cache-control: no-cache\r\n'
        b'Pragma: no-cache\r\n'
        b'Expires: 0\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class RedisImageStream:
    def __init__(self, camera, boxes, field):
        self.camera = camera
        self.boxes = boxes
        self.field = field.encode('utf-8')

    async def __aiter__(self):
        while True:
            frame = await self.latest()
            if frame is None:
                break
            yield frame



    async def latest(self):
        ''' Gets latest from camera and model '''
        # await asyncio.sleep(0.5)
        # return technical_difficulties
        # print('before pipeline', red)
        # async with red.pipeline() as p:
        p = red.pipeline()
        p.xrevrange(self.camera, count=1)  # Latest frame
        p.xrevrange(self.boxes, count=1)   # Latest boxes
        print('before execute', red, p)
        cmsg, bmsg = await p.execute()
        print('cmsg', len(cmsg) if cmsg else type(cmsg), len(bmsg) if bmsg else type(bmsg))

        # if cmsg:
        #     return cmsg[0][1][self.field]
        if cmsg:
            last_id = cmsg[0][0].decode('utf-8')
            label = f'{self.camera}:{last_id}'
            data = io.BytesIO(cmsg[0][1][self.field])
            img = Image.open(data)
            if bmsg:
                boxes = np.fromstring(bmsg[0][1]['boxes'.encode('utf-8')][1:-1], sep=',')
                label += ' people: {}'.format(bmsg[0][1]['people'.encode('utf-8')].decode('utf-8'))
                for box in range(int(bmsg[0][1]['people'.encode('utf-8')])):  # Draw boxes
                    x1, y1, x2, y2 = boxes[box*4:(box + 1)*4]
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(((x1, y1), (x2, y2)), width=5, outline='red')
            arr = np.array(img)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cv2.putText(arr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', arr)
            return img.tobytes()
        
        await asyncio.sleep(0.5)
        return technical_difficulties
