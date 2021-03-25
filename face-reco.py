import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mtcnn.mtcnn import MTCNN
import glob
import os


def extract_face(filename, required_size=(224, 244)):
    # img = Image.open(filename)
    img = plt.imread(filename)
    print(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    if not results:
        os.remove(filename)
        return None
    x1, y1, width, height = results[0]['box']
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    # face = img.crop((x1, y1, x1 + width, y1 + height))
    image = Image.fromarray(face)
    image = image.resize(required_size)
    image.save(filename)
    face_array = np.asarray(image)
    return face_array


images = [extract_face(file) for file in glob.glob("./images/Mask/*.jpg")]
# img = extract_face('./images/001.jpg')
#
# plt.imshow(img)
# plt.show()
