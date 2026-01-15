import io
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# ensure 'backend' is on sys.path so 'app' package can be imported
proj_root = Path(__file__).resolve().parent.parent
backend_dir = proj_root / 'backend'
sys.path.insert(0, str(backend_dir))

from app import main

# create a synthetic RGB image with the expected size
img = Image.fromarray((np.random.rand(*main.IMG_SIZE, 3) * 255).astype('uint8'))
buf = io.BytesIO()
img.save(buf, format='PNG')
image_bytes = buf.getvalue()

# preprocess
input_arr = main.preprocess_image(image_bytes)

print('Input shape:', input_arr.shape)

# run CML (classical) prediction
try:
    cml_preds = main.cml_model.predict(input_arr)
    cml_class = int(np.argmax(cml_preds))
    cml_conf = float(np.max(cml_preds))
    print('CNN prediction:', main.idx_to_class.get(cml_class, str(cml_class)), ' confidence:', round(cml_conf * 100, 2))
except Exception as e:
    print('CNN prediction error:', e)

# run QML prediction
try:
    if main.qml_model is None:
        print('QML model not available:', getattr(main, 'qml_load_error', 'unknown'))
    else:
        # QML model expects a 1D feature vector of length 128 (see model config). Use
        # a random vector here for demonstration; replace with proper feature extraction.
        qml_input = np.random.rand(1, 128).astype('float32')
        qml_preds = main.qml_model.predict(qml_input)
        qml_class = int(np.argmax(qml_preds))
        qml_conf = float(np.max(qml_preds))
        print('QML prediction:', main.idx_to_class.get(qml_class, str(qml_class)), ' confidence:', round(qml_conf * 100, 2))
except Exception as e:
    print('QML prediction error:', e)
