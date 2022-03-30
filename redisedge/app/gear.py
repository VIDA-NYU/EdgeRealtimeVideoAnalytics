# A Redis gear for orchestrating realtime video analytics
import os
import io
import sys
sys.path.append('/opt/redislabs/lib/modules/python3')
sys.path.append('/app')
os.chdir('/app')

print("Ran gears", flush=True)

import json
# import cv2
import redisAI
import numpy as np
from time import time
from PIL import Image

from redisgears import executeCommand as execute

import models


class SimpleMovingAverage(object):
    ''' Simple moving average '''
    def __init__(self, value=0.0, count=7):
        '''
        @value - the initialization value
        @count - the count of samples to keep
        '''
        self.count = int(count)
        self.current = float(value)
        self.samples = [self.current] * self.count

    def __str__(self):
        return str(round(self.current, 3))

    def add(self, value):
        ''' Adds the next value to the average '''
        v = float(value)
        self.samples.insert(0, v)
        o = self.samples.pop()
        self.current = self.current + (v-o)/self.count

class Profiler(object):
    ''' Mini profiler '''
    names = []  # Steps names in order
    data = {}   # ... and data
    last = None
    def __init__(self):
        pass

    def __str__(self):
        s = ''
        for name in self.names:
            s = '{}{}:{}, '.format(s, name, self.data[name])
        return(s[:-2])

    def __delta(self):
        ''' Returns the time delta between invocations '''
        now = time()*1000       # Transform to milliseconds
        if self.last is None:
            self.last = now
        value = now - self.last
        self.last = now
        return value

    def start(self):
        ''' Starts the profiler '''
        self.last = time()*1000

    def add(self, name):
        ''' Adds/updates a step's duration '''
        value = self.__delta()
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def assign(self, name, value):
        ''' Assigns a step with a value '''
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def get(self, name):
        ''' Gets a step's value '''
        return self.data[name].current

    def publish_metrics(self, video_name, ref_msec):
        # Record profiler steps
        for name in self.names:
            current = self.data[name].current
            execute('TS.ADD', f'{video_name}:prf_{name}', ref_msec, current)
        self.add('store')


class Downsampler:
    # Globals for downsampling
    _mspf = 1000 / 10.0      # Msecs per frame (initialized with 10.0 FPS)
    _next_ts = 0             # Next timestamp to sample a frame

    def filter(self, x):
        ''' Drops input frames to match FPS '''
        ts, _ = map(int, str(x['streamId']).split('-'))         # Extract the timestamp part from the message ID
        sample_it = self._next_ts <= ts
        if sample_it:                                           # Drop frames until the next timestamp is in the present/past
            self._next_ts = ts + self._mspf
            log(f'!!! waiting {self._next_ts} {ts}')
        else:
            # print('!!! done', self._next_ts, ts)
            log(f'!!! done {self._next_ts} {ts}')
        # execute('TS.INCRBY', f'camera:0:dropping_frame', (int(sample_it)+1)*3, 'RESET', 1)
        return sample_it

    def update(self, mspf):
        self._mspf = mspf * 1.05  # A little extra leg room


'''
The profiler is used first and foremost for keeping track of the total (average) time it takes to process
a frame - the information is required for setting the FPS dynamically. As a side benefit, it also provides
per step metrics.
'''
prf = Profiler()
downsample = Downsampler()


class LazyRef:
    _current = None
    def __init__(self, get, *a, **kw):
        self.get, self.a, self.kw = get, a, kw

    @property
    def current(self):
        m = self._current
        if m is None:
            m = self._current = self.get(*self.a, **self.kw)
        return m

def run_yolo_model(name='tinyyolov3'):
    # ref = LazyRef(models.get_model, name)
    model = models.models[name]
    def run(x):
        # execute('TS.INCRBY', f'{name}:about_to_model', 1, 'RESET', 1)
        # model = ref.current
        prf.start()        # Start a new profiler iteration
        outputs = [[]]
        # # Read the image from the stream's message
        img = np.array(Image.open(io.BytesIO(x['image'])))
        # print(img.shape)
        prf.add('read')
        inputs = model.preprocess(img)  # TODO: should accept batch of images
        inputs = [redisAI.createTensorFromBlob('FLOAT', i.shape, i) for i in inputs]
        prf.add('preprocess')
        # outputs = model.predict(*inputs)
        
        modelRunner = redisAI.createModelRunner(f'{name}:model')
        for k, i in zip(model.INPUTS, inputs):
            redisAI.modelRunnerAddInput(modelRunner, k, i)
        for k in model.OUTPUTS:
            redisAI.modelRunnerAddOutput(modelRunner, k)
        outputs = redisAI.modelRunnerRun(modelRunner)
        prf.add('model')

        outputs = model.postprocess(*outputs)
        prf.add('postprocess')

        # execute('TS.INCRBY', f'{name}:finished_model', 1, 'RESET', 1)
        
        return x['streamId'], outputs[0]
    return run


def store_results(camera_name='camera:0', model_name='yolo', key='boxes'):
    def store(x):
        ''' Stores the results in Redis Stream and TimeSeries data structures '''
        ref_id, boxes = x

        # Store the output in its own stream
        res_id = execute(
            'XADD', f'{camera_name}:{model_name}', 
            'MAXLEN', '~', 1000, '*', 
            'ref', ref_id, key, boxes)

        # Adjust mspf to the moving average duration
        res_msec = int(str(res_id).split('-')[0])
        ref_msec = int(str(ref_id).split('-')[0])
        prf.assign('total', res_msec - ref_msec)
        downsample.update(prf.get('total'))
        prf.publish_metrics(camera_name, ref_msec)
    return store


def clock_fps(input_name, fps_name='fps'):
    def clock_fps(x):
        execute('TS.INCRBY', f'{input_name}:{fps_name}', 1, 'RESET', 1)
        return x
    return clock_fps

def main(name='camera:0', model_name='tinyyolov3', gears_name='StreamReader'):
    # Create and register a gear that for each message in the stream
    gb = GearsBuilder(gears_name)
    gb.map(clock_fps(name, 'in_fps'))
    gb.filter(lambda x: downsample.filter(x))  # Filter out high frame rate
    gb.map(clock_fps(name, 'ds_fps'))
    gb.map(run_yolo_model(model_name))              # Run the model
    gb.map(store_results(name, model_name))         # Store the results
    gb.map(clock_fps(name, 'out_fps'))
    gb.register(name)
