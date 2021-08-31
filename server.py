import os
import gc
import json
import numpy as np
import random
import glob
import torch
import torchvision
import torchvision.transforms as T
import tornado.web
import tornado.ioloop
import mysql.connector
import pydicom
import face_recognition

from PIL import Image
from shutil import move
from typing import Optional, Awaitable
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PORT = 8000
MYSQL = {
    'host': 'localhost',
    'user': 'root',
    'passwd': '',
    'database': 'covid-19-detection',
}
# "negative", "typical", "indeterminate", "atypical"
NUM_CLASSES = 4
MODEL_PATH = 'model/best-checkpoint.bin'


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return b, g, r


# https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


# https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
def resize(data, size=None, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(data)
    if size is not None:
        if keep_ratio:
            im.thumbnail((size, size), resample)
        else:
            im = im.resize((size, size), resample)
    return im


def convert_dicom_to_image(path, voi_lut=True, fix_monochrome=True,
                           size=None, keep_ratio=False, resample=Image.LANCZOS,
                           jpg=True):
    xray = read_xray(path, voi_lut, fix_monochrome)
    image = resize(xray, size, keep_ratio, resample)
    path_name, extension = os.path.splitext(path)
    if jpg == 1:
        image.save(path_name + '.jpg')
    else:
        image.save(path_name + '.png')


def recognize(username, path):
    user_face = face_recognition.load_image_file(f'recognize/faces/{username}')
    user_face_encoding = face_recognition.face_encodings(user_face)[0]
    unknown_face = face_recognition.load_image_file(path)
    unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
    results = face_recognition.compare_faces([user_face_encoding], unknown_face_encoding)
    if results[0]:
        return True
    else:
        return False


def get_model(checkpoint_path=None, pretrained=False):
    model = FasterRCNNDetector(pretrained=pretrained)
    # Load the trained weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        gc.collect()
    return model.cuda()


def predict(path):
    img = Image.open(path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    model = get_model(MODEL_PATH)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.cuda()
            img = img.cuda()
        else:
            device = torch.device("cpu")
            model.cpu()
            img = img.cpu()
        outputs = model([img])
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    return outputs


def empty_cache():
    torch.cuda.empty_cache()


class FasterRCNNDetector(torch.nn.Module):

    def __init__(self, pretrained=False, **kwargs):
        super(FasterRCNNDetector, self).__init__()
        # load pre-trained model incl. head
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                          pretrained_backbone=pretrained)
        # get number of input features for the classifier custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    def forward(self, images, targets=None):
        return self.model(images, targets)


class BaseRequestHandler(tornado.web.RequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Credentials", "true")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')

    # vue一般需要访问options方法， 如果报错则很难继续，所以只要通过就行了，当然需要其他逻辑就自己控制。
    # 因为vue访问时，会先执行一次预加载，直接放过就好了
    def options(self):
        # 这里的状态码一定要设置200
        self.set_status(200)
        self.finish()


class IndexRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        self.render("index.html")


class LoginRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        username = data['username']
        password = data['password']
        print('username %s, password %s' % (username, password))
        db = mysql.connector.connect(
            host=MYSQL['host'],
            user=MYSQL['user'],
            passwd=MYSQL['passwd'],
            database=MYSQL['database']
        )
        cursor = db.cursor()
        cursor.execute(f'select count(*) from users where username = "{username}" and password = "{password}";')
        result = cursor.fetchone()
        cursor.close()
        db.close()
        if result[0] == 1:
            response = {
                'code': 0
            }
        else:
            response = {
                'code': 1
            }
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


class RecognitionRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        files = self.request.files['recognize']
        if len(files) > 0:
            file = files[0]
            file_bytes = open(f'recognize/{file.filename}', 'wb')
            file_bytes.write(file.body)
            file_bytes.close()
            if file.filename.endswith("png") or file.filename.endswith("jpg"):
                try:
                    result = recognize(file.filename, f'recognize/{file.filename}')
                    if result:
                        response = {
                            'code': 0
                        }
                    else:
                        response = {
                            'code': 1
                        }
                except IndexError:
                    response = {
                        'code': 1
                    }
            else:
                response = {
                    'code': 1
                }
            self.set_header('Content-type', 'application/json')
            self.write(bytes(json.dumps(response), 'utf-8'))


class DetectRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        print(self.request)
        files = self.request.files['detect']
        if len(files) > 0:
            file = files[0]
            path_name, extension = os.path.splitext(file.filename)
            if extension == '.dcm':
                file_bytes = open(f'detect/{file.filename}', 'wb')
                file_bytes.write(file.body)
                file_bytes.close()
                try:
                    convert_dicom_to_image(f'detect/{file.filename}', jpg=False)
                    convert_dicom_to_image(f'detect/{file.filename}', size=256)
                    outputs = predict(f'detect/{path_name}.jpg')
                    boxes = outputs[0]['boxes'].cpu().numpy().astype(np.float64).tolist()
                    labels = outputs[0]['labels'].cpu().numpy().astype(np.int32).tolist()
                    scores = outputs[0]['scores'].cpu().numpy().astype(np.float64).tolist()
                    empty_cache()
                    db = mysql.connector.connect(
                        host=MYSQL['host'],
                        user=MYSQL['user'],
                        passwd=MYSQL['passwd'],
                        database=MYSQL['database']
                    )
                    cursor = db.cursor()
                    cursor.execute(f'insert into radiographs (is_marked) values (1);')
                    db.commit()
                    cursor.execute('select LAST_INSERT_ID();')
                    result = cursor.fetchone()
                    cursor.close()
                    os.rename(f'detect/{file.filename}', f'detect/{result[0]}{extension}')
                    os.rename(f'detect/{path_name}.png', f'detect/{result[0]}.png')
                    move(f'detect/{result[0]}{extension}', f'unmarked/{result[0]}{extension}')
                    move(f'detect/{result[0]}.png', f'show/{result[0]}.png')
                    os.remove(f'detect/{path_name}.jpg')
                    response = {
                        'code': 0,
                        'id': result[0],
                        'boxes_256': boxes,
                        'labels': labels,
                        'scores': scores
                    }
                except FileNotFoundError:
                    files = glob.glob('detect/*')
                    for f in files:
                        os.remove(f)
                    response = {
                        'code': 1
                    }
                except RuntimeError:
                    files = glob.glob('detect/*')
                    for f in files:
                        os.remove(f)
                    response = {
                        'code': 1
                    }
            else:
                response = {
                    'code': 1
                }
        else:
            response = {
                'code': 1
            }
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


class PreviousRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        try:
            page_id = int(self.get_argument('id'))
            db = mysql.connector.connect(
                host=MYSQL['host'],
                user=MYSQL['user'],
                passwd=MYSQL['passwd'],
                database=MYSQL['database']
            )
            cursor = db.cursor()
            cursor.execute(
                f'select id from radiographs where id < {page_id} and is_marked = 1 order by id desc limit 1;'
            )
            result = cursor.fetchone()
            cursor.close()
            db.close()
            if result is not None:
                response = {
                    'code': 0,
                    'id': result[0]
                }
            else:
                response = {
                    'code': 0,
                    'id': -1
                }
        except ValueError:
            response = {
                'code': 1,
            }
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


class NextRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        try:
            page_id = int(self.get_argument('id'))
            db = mysql.connector.connect(
                host=MYSQL['host'],
                user=MYSQL['user'],
                passwd=MYSQL['passwd'],
                database=MYSQL['database']
            )
            cursor = db.cursor()
            cursor.execute(
                f'select id from radiographs where id > {page_id} and is_marked = 1 order by id limit 1;'
            )
            result = cursor.fetchone()
            cursor.close()
            db.close()
            if result is not None:
                response = {
                    'code': 0,
                    'id': result[0]
                }
            else:
                response = {
                    'code': 0,
                    'id': -1
                }
        except ValueError:
            response = {
                'code': 1,
            }
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


class DrawRequestHandler(BaseRequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        page_id = data['id']
        class_id = data['class']
        boxes = data['boxes']
        print('id %s, boxes %s' % (page_id, boxes))
        db = mysql.connector.connect(
            host=MYSQL['host'],
            user=MYSQL['user'],
            passwd=MYSQL['passwd'],
            database=MYSQL['database']
        )
        cursor = db.cursor()
        cursor.execute(
            f'update radiographs set class = "{class_id}" boxes = "{boxes}" is_marked = 0 where id = {page_id};'
        )
        db.commit()
        # Previous
        cursor.execute(
            f'select id from radiographs where id < {page_id} and is_marked = 1 order by id desc limit 1;'
        )
        previous_result = cursor.fetchone()
        if previous_result is not None:
            response = {
                'code': 0,
                'id': previous_result[0]
            }
        else:
            # Next
            cursor.execute(
                f'select id from radiographs where id > {page_id} and is_marked = 1 order by id limit 1;'
            )
            next_result = cursor.fetchone()
            if next_result is not None:
                response = {
                    'code': 0,
                    'id': next_result[0]
                }
            else:
                response = {
                    'code': 0,
                    'id': -2
                }
        cursor.close()
        db.close()
        self.set_header('Content-type', 'application/json')
        self.write(bytes(json.dumps(response), 'utf-8'))


if __name__ == "__main__":
    app = tornado.web.Application([
        ("/", IndexRequestHandler),
        ("/login", LoginRequestHandler),
        ("/recognize", RecognitionRequestHandler),
        ("/detect", DetectRequestHandler),
        ("/previous", PreviousRequestHandler),
        ("/next", NextRequestHandler),
        ("/draw", DrawRequestHandler),
        ("/show/(.*)", tornado.web.StaticFileHandler, {'path': 'show'})
    ])

    app.listen(PORT)
    print("Listening on port %s" % PORT)
    tornado.ioloop.IOLoop.instance().start()
