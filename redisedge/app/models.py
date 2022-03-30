import os
import cv2
import numpy as np
from PIL import Image


pjoin = os.path.join

asset_dir = os.path.abspath(os.path.join(__file__, '..', 'assets'))

def read_labels(fname):
    with open(fname, 'r') as f:
        return [l.strip() for l in f.read().splitlines()]


VOC_LABELS = read_labels(pjoin(asset_dir, 'VOC_labels.txt'))
COCO_LABELS = read_labels(pjoin(asset_dir, 'COCO_labels.txt'))

providers = [
    # 'CUDAExecutionProvider',
    'CPUExecutionProvider'
]


class Onnx:
    path = None
    download_url = None

    def __init__(self):
        path = self.download_weights()

        import onnxruntime
        self.sess = sess = onnxruntime.InferenceSession(path, providers=providers)
        self.input_names = [i.name for i in sess.get_inputs()]

    def __call__(self, img):
        return self.postprocess(*self.predict(*self.preprocess(img)))

    def predict(self, *inputs):
        return self.sess.run(None, {k: np.asarray(x) for k, x in zip(self.input_names, inputs)})

    @classmethod
    def preprocess(cls, x): return x

    @classmethod
    def postprocess(cls, x): return x

    @classmethod
    def download_weights(cls):
        path = pjoin(asset_dir, cls.path or cls.download_url.split('/')[-1])
        if not os.path.isfile(path) and cls.download_url:
            download_file(cls.download_url, path)
        return path

    


class YoloBase(Onnx):
    anchors = []
    conf_threshold = 0.5
    input_size = (416, 416)
    nms_threshold = 0.3
    norm = True

    @classmethod
    def preprocess(cls, img):
        img = img[:,:,:3]
        img = cv2.resize(img, cls.input_size)
        img = np.transpose(img, [2, 0, 1])
        if cls.norm:
            img = img / 255.
        return [img[None].astype('float32')]

    @classmethod
    def postprocess(cls, *outs_batch):
        return [
            nms(list(cls._postprocess(*outs)), cls.nms_threshold)
            for outs in zip(*outs_batch)
        ]


class Yolov2(YoloBase):
    '''A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a single network evaluation, 
    enabling it to be more than 1000x faster than R-CNN and 100x faster than Faster R-CNN. This model is trained with COCO dataset and 
    contains 80 classes.
    '''
    download_url = 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx'
    labels = COCO_LABELS
    anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)])
    norm = True
    conf_threshold = 0.3
    INPUTS = ['input.1']
    OUTPUTS = ['218']
    INPUT_DTYPES = ['FLOAT']

    @classmethod
    def _postprocess(cls, Y):
        labels, anchors, conf_threshold = cls.labels, cls.anchors, cls.conf_threshold
        channel_stride = (len(labels) + 5)
        nch, nx, ny = Y.shape
        assert nch == len(anchors) * channel_stride, (
            "Incorrect output size. Check label and anchor dims: {} * ({} + 5) != {}".format(len(anchors), len(labels), nch))
        for cx in range(nx):
            for cy in range(ny):
                yij = Y[:, cy, cx]
                for b in range(len(anchors)):
                    tx, ty, tw, th, tc, *cls_scores = yij[b * channel_stride:(b + 1) * channel_stride].tolist()
                    confidence = sigmoid(tc)
                    if confidence < conf_threshold:
                        continue
                    i_max = np.argmax(cls_scores)
                    top_score = cls_scores[i_max]
                    if top_score < conf_threshold:
                        continue

                    x = (cx + sigmoid(tx)) / nx
                    y = (cy + sigmoid(ty)) / ny
                    w = np.exp(tw) * anchors[b,0] / nx
                    h = np.exp(th) * anchors[b,1] / ny

                    yield {
                        'x': x - w/2, 
                        'y': y - h/2, 
                        'w': w, 'h': h, 
                        'label': labels[i_max], 
                        'top_score': top_score, 
                        'confidence': confidence, 
                    }


class TinyYolov2(Yolov2):
    '''A real-time CNN for object detection that detects 20 different classes (VOC). A smaller version of the more complex full YOLOv2 network.
    '''
    download_url = 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx'
    labels = VOC_LABELS
    norm = False
    INPUTS = ['image']
    OUTPUTS = ['grid']
    


class Yolov3(YoloBase):
    '''A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than 
    YOLOv2 but still very fast. As accurate as SSD but 3 times faster.
    '''
    download_url = 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx'
    labels = COCO_LABELS
    INPUTS = ['input_1', 'image_shape']
    OUTPUTS = ['yolonms_layer_1/ExpandDims_1:0', 'yolonms_layer_1/ExpandDims_3:0', 'yolonms_layer_1/concat_2:0']

    @classmethod
    def preprocess(cls, img):
        preprocessed = super().preprocess(img)
        shape = np.asarray(img.shape[-2:][::-1], dtype='float32').reshape(1, 2)
        return preprocessed + [shape]

    @classmethod
    def _postprocess(cls, boxes, scores, indices):
        ih, iw = cls.input_size
        for ibatch, icls, ibox in indices:
            y, x, y2, x2 = boxes[ibatch, ibox]
            score = scores[ibatch, icls, ibox]
            if score < cls.conf_threshold:
                continue

            yield {
                'x': x/iw, 
                'y': y/ih, 
                'w': (x2-x)/iw, 
                'h': (y2-y)/ih, 
                'label': cls.labels[icls], 
                'top_score': 1, 
                'confidence': score, 
            }


class TinyYolov3(Yolov3):
    '''A smaller version of YOLOv3 model.'''
    download_url = 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx'
    labels = COCO_LABELS
    INPUTS = ['input_1', 'image_shape']
    OUTPUTS = ['yolonms_layer_1', 'yolonms_layer_1:1', 'yolonms_layer_1:2']
    def _postprocess(cls, boxes, scores, indices):
        return super()._postprocess(boxes, scores, indices[0])


class Yolov4(YoloBase):
    '''Optimizes the speed and accuracy of object detection. Two times faster than EfficientDet. It improves 
    YOLOv3's AP and FPS by 10% and 12%, respectively, with mAP50 of 52.32 on the COCO 2017 dataset and FPS 
    of 41.7 on a Tesla V100.
    '''
    download_url = 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx'
    labels = COCO_LABELS
    anchors = np.array([
        [[12,16], [19,36], [40,28]], 
        [[36,75], [76,55], [72,146]], 
        [[142,110], [192,243], [459,401]],
    ])
    xyscale = [1.2, 1.1, 1.05]
    threshold = 0.6
    INPUTS = ['input_1:0']
    OUTPUTS = ['Identity:0', 'Identity_1:0', 'Identity_2:0']

    @classmethod
    def postprocess(cls, *outs):
        labels, threshold = cls.labels, cls.conf_threshold

        for anchors, Y, xyscale in zip(cls.anchors, outs, cls.xyscale):
            nx, ny, nanch, nfeat = Y.shape
            assert nanch == len(anchors), f"Wrong number of anchors {len(anchors)} != {nanch}"
            assert nfeat == len(labels) + 5, f"Wrong number of labels {len(labels)} + 5 != {nfeat}"
            for cx in range(nx):
                for cy in range(ny):
                    for b in range(len(anchors)):
                        tx, ty, tw, th, tc, *cls_scores = Y[cy, cx, b]
                        confidence = tc#sigmoid(tc)
                        if confidence < threshold:
                            continue
                        i_max = np.argmax(cls_scores)
                        top_score = cls_scores[i_max]
                        if top_score < threshold:
                            continue

                        x = (cx + (sigmoid(tx) - 1/2) * xyscale + 1/2) / nx
                        y = (cy + (sigmoid(ty) - 1/2) * xyscale + 1/2) / ny
                        w = np.exp(tw) * anchors[b,0] / nx
                        h = np.exp(th) * anchors[b,1] / ny

                        yield {
                            'x': x - w/2, 
                            'y': y - h/2, 
                            'w': w, 'h': h, 
                            'label': labels[i_max], 
                            'top_score': top_score, 
                            'confidence': confidence, 
                        }




def sigmoid(x):
  return 1 / (1 + np.exp(-x))



def nms(boxes, threshold=0.3):
    n_active = len(boxes)
    active = [True]*len(boxes)
    boxes = sorted(boxes, key=lambda b: b['confidence'], reverse=True)
    
    for i, bA in enumerate(boxes):
        if not n_active: break
        if not active[i]: continue
        for j, bB in enumerate(boxes[i+1:], i+1):
            if not active[j]: continue
            if IoU(bA, bB) > threshold:
                active[j] = False
                n_active -= 1
    # print('keep', [b.confidence for i, b in enumerate(boxes) if active[i]], 'remove', [b.confidence for i, b in enumerate(boxes) if not active[i]])
    return [b for i, b in enumerate(boxes) if active[i]]


def IoU(a, b):
    areaA = a['w'] * a['h']
    areaB = b['w'] * b['h']
    if areaA <= 0 or areaB <= 0: return 0
    intersectionArea = (
        max(min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']), 0) * 
        max(min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x']), 0))
    return intersectionArea / (areaA + areaB - intersectionArea)






models = {
    'tinyyolov2': TinyYolov2,
    'yolov2': Yolov2,
    'yolov3': Yolov3,
    'tinyyolov3': TinyYolov3,
    'yolov4': Yolov4,
}



def get_model(name):
    model = models[name.lower()]()
    return model

def show(*names):
    import inspect
    for name, m in [(n, models[n]) for n in names] or models.items():
        print('-'*20)
        print(f'{name} : {m.__name__}')
        print(inspect.cleandoc(m.__doc__), '\n')
        if getattr(m, 'download_url', None):
            print(f'\tDownload url:', m.download_url)
        if getattr(m, 'labels', None):
            print(f'\tLabels: ({len(m.labels)})', m.labels)
        print()
        print()


def download_file(url, local_filename=None):
    import requests
    local_filename = local_filename or url.split('/')[-1]
    print(f"Downloading {local_filename} to {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    print('download finished.')
    return local_filename
