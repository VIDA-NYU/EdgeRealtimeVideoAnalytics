# RedisEdge realtime video analytics web server
import os
import io
import json
import time
import asyncio
# from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw

# import redis
from redis import asyncio as aioredis
# import aioredis
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



@app.get('/')
def index():
    return 'Hi!'

@app.get('/ping')
def ping():
    return PlainTextResponse(':)')


@app.get('/webcam')
def webcam(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})



@app.websocket('/video/push')
async def video_input(websocket: WebSocket, name='camera:0', field='image', maxlen=10000):
    await websocket.accept()
    try:
        i = 0
        while True:
            data = await websocket.receive_bytes()
            _id = await red.xadd(name, { 'index': i, field: data }, maxlen=maxlen)
            i += 1
    except WebSocketDisconnect:
        print('Video Input Disconnected')



@app.get('/video/pull')
def video_feed(device_name='camera:0'):
    async def stream():
        while True:
            p = red.pipeline(transaction=True)
            p.xrevrange(device_name, count=1)  # Latest frame
            # p.xrevrange(boxes_name, count=1)   # Latest boxes
            cmsg, = await p.execute()

            f = cmsg[0][1]['image'.encode('utf-8')] if cmsg else None
            if f is None:
                yield img_replace_frame(blank)
                await asyncio.sleep(1)
                continue

            yield img_replace_frame(f)
            await asyncio.sleep(0.01)

    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')



@app.websocket('/boxes/pull')
async def video_feed(websocket: WebSocket, device_name='camera:0', model='tinyyolov2'):
    boxes_name = f'{device_name}:boxes'
    await websocket.accept()
    try:
        i = 0
        while True:
            p = red.pipeline(transaction=True)
            p.xrevrange(boxes_name, count=1)
            bmsg, = await p.execute()
            if bmsg:
                websocket.send_json({
                    'id': bmsg[0][0].decode('utf-8'),
                    'boxes': json.loads(bmsg[0][1]['boxes'.encode('utf-8')]),
                })
    except WebSocketDisconnect:
        print('Video Input Disconnected')



@app.get('/video+boxes/pull')
def video_feed(device_name='camera:0', model='tinyyolov2'):
    boxes_name = f'{device_name}:boxes'
    async def stream():
        while True:
            p = red.pipeline(transaction=True)
            p.xrevrange(device_name, count=1)  # Latest frame
            p.xrevrange(boxes_name, count=1)   # Latest boxes
            cmsg, bmsg = await p.execute()

            f = draw_boxes_on_image_bytes(cmsg, bmsg, device_name)
            if f is None:
                yield img_replace_frame(technical_difficulties)
                await asyncio.sleep(1)
                continue

            yield img_replace_frame(f)
            await asyncio.sleep(0.01)

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


def draw_boxes_on_image_bytes(cmsg, bmsg, label):
    if not cmsg: return

    img = Image.open(io.BytesIO(cmsg[0][1]['image'.encode('utf-8')]))
    W, H = img.size
    label=f"{label}:{cmsg[0][0].decode('utf-8')}"
    if bmsg:
        m = bmsg[0][1]
        boxes = json.loads(m['boxes'.encode('utf-8')])
        # boxes = np.fromstring(m['boxes'.encode('utf-8')][1:-1], sep=',')
        # n_ppl = m['people'.encode('utf-8')]
        # label += ' people: {}'.format(n_ppl.decode('utf-8'))
        for box in boxes:  # Draw boxes
            x, y, w, h = box['x']*W, box['y']*H, box['w']*W, box['h']*H
            draw = ImageDraw.Draw(img)
            draw.rectangle(((x, y), (x+w, y+h)), width=5, outline='red')
            label +=  ' '+box['label']
    arr = np.array(img)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    cv2.putText(arr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
    ret, img = cv2.imencode('.jpg', arr)
    return img.tobytes()