import requests, io
from PIL import Image
import numpy as np

def post(url):
    img = Image.fromarray((np.random.rand(160,160,3)*255).astype('uint8'))
    buf = io.BytesIO(); img.save(buf, format='JPEG'); buf.seek(0)
    files = {'file': ('test.jpg', buf, 'image/jpeg')}
    try:
        r = requests.post(url, files=files, timeout=10)
        print('URL', url, 'STATUS', r.status_code, 'BODY', r.text)
    except Exception as e:
        print('URL', url, 'ERROR', e)

post('http://127.0.0.1:8000/predict-cnn')
post('http://127.0.0.1:8000/predict-qml')
