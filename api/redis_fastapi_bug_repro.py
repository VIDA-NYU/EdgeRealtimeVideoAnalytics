import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from redis import asyncio as aioredis

app = FastAPI()
red = aioredis.from_url('redis://redisedge:6379')

with open('data/random.jpg', 'rb') as f:
    placeholder_image = f.read()

@app.get('/')
def index():
    return HTMLResponse('<img src="/video" width="600" height="400" />')

@app.get('/push')
async def add_image():
    return await red.xadd('camera:bug-test', { 'image': placeholder_image })

@app.get('/video')
def video_feed(name='camera:bug-test',field='image'):
    async def stream():
        while True:
            # query image
            p = red.pipeline(transaction=True).xrevrange(name, count=1)
            print('before execute')
            cmsg, = await p.execute()
            print('after execute', len(cmsg))

            # serve image
            if not cmsg:
                break
            yield format_frame(cmsg[0][1][field.encode('utf-8')])
            await asyncio.sleep(0.01)

    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')


def format_frame(frame):
    return (
        b'--frame\r\n'
        b'Pragma-directive: no-cache\r\n'
        b'Cache-directive: no-cache\r\n'
        b'Cache-control: no-cache\r\n'
        b'Pragma: no-cache\r\n'
        b'Expires: 0\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
