from flask import Flask, request
from flask_cors import CORS
from werkzeug import secure_filename
from faceRecognition.vgg_face import findMostSimilar
import pathlib
import os
import json
from PIL import Image
from subprocess import call

# Create folder uploads if it does not exist

COMBINED_FOLDER = os.path.join(os.getcwd(), 'Images/combined')
pathlib.Path(COMBINED_FOLDER).mkdir(parents=True, exist_ok=True)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Images/uploads')
pathlib.Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

GENERATED_FOLDER = os.path.join(os.getcwd(), 'Images/generated')
pathlib.Path(GENERATED_FOLDER).mkdir(parents=True, exist_ok=True)

PHOTOS_FOLDER = os.path.join(os.getcwd(), 'Images/photos')
pathlib.Path(PHOTOS_FOLDER).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def doubleImage(path):
    img = Image.open(path)

    new_im = Image.new('RGB', (256 * 2, 256))
    new_im.paste(img, (0, 0))
    new_im.paste(img, (256, 0))
    new_path = os.path.join(COMBINED_FOLDER, os.path.basename(path))
    new_im.save(new_path)
    return new_path


def saveImage():
    f = request.files['file']
    filename = secure_filename(f.filename)
    pathFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(pathFile)
    return pathFile


def generateSketch(pathFile):
    combinedPath = doubleImage(pathFile)

    call([
        "python", "pix2pix/pix2pix.py", "--mode", "test", "--output_dir",
        "pix2pix/test", "--input_dir", COMBINED_FOLDER, "--checkpoint",
        "pix2pix/photos_train"
    ])
    filename = os.path.basename(pathFile)
    targetName = os.path.splitext(filename)[0] + '-outputs' + os.path.splitext(
        filename)[1]

    targetPath = os.path.join(GENERATED_FOLDER, filename)
    os.rename("pix2pix/test/images/" + targetName, targetPath)
    os.remove(combinedPath)

    return targetPath


@app.route('/index')
def hello_world():
    return 'Hello World'


@app.route('/pix2pix', methods=['POST'])
def pix2pix():
    pathFile = saveImage()
    sketchPath = generateSketch(pathFile)
    mostSimilar = findMostSimilar(sketchPath, 3)
    print(mostSimilar)
    mostSimilar = [
        (os.path.join(PHOTOS_FOLDER, k[0]), k[1]) for k in mostSimilar
    ]
    return json.dumps((sketchPath, mostSimilar))
    # return json.dumps(sketchPath)


@app.route('/findSimilar', methods=['POST'])
def findSimilar():
    pathFile = saveImage()
    mostSimilar = [
        os.path.join(PHOTOS_FOLDER, k) for k in findMostSimilar(pathFile, 3)
    ]
    return json.dumps(mostSimilar)


if __name__ == '__main__':
    app.run()
