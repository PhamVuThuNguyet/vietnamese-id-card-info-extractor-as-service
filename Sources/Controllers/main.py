import os
import re
import pytesseract
from pytesseract import Output
import cv2
import Sources.Controllers.config as cfg
import numpy as np
import yolov5
from PIL import Image
from Sources import app
from Sources.Controllers import utils, rekognition, database_management
from fastapi import UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

""" ---- Setup ---- """
# Constant
OFFSET = 30
NMS_THRESHOLD = 0.7

# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR

""" Recognizion detected parts in ID """
config = Cfg.load_config_from_name('vgg_seq2seq')  # OR vgg_transformer -> acc || vgg_seq2seq -> time
config['weights'] = cfg.OCR_MODEL_PATH
config['cnn']['pretrained'] = False
config['device'] = cfg.DEVICE
config['predictor']['beamsearch'] = False
detector = Predictor(config)


@app.post("/upload")
async def upload(frontFile: UploadFile = File(...), backFile: UploadFile = File(...)):
    try:
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)

        input_images_list = os.listdir(UPLOAD_FOLDER)
        if input_images_list is not None:
            for uploaded_image in input_images_list:
                os.remove(os.path.join(UPLOAD_FOLDER, uploaded_image))

        front_file_location = f'{UPLOAD_FOLDER}/{frontFile.filename}'
        back_file_location = f'{UPLOAD_FOLDER}/{backFile.filename}'

        """FRONT FILE"""
        front_data = await frontFile.read()
        with open(front_file_location, 'wb') as f:
            f.write(front_data)

        """BACK FILE"""
        back_data = await backFile.read()
        with open(back_file_location, 'wb') as f:
            f.write(back_data)

        # Validating file
        input_files = os.listdir(UPLOAD_FOLDER)[:2]
        for input_file in input_files:
            if input_file == 'NULL':
                os.remove(os.path.join(UPLOAD_FOLDER, input_file))
                error = "No file selected!"
                return JSONResponse(status_code=403, content={"errorCode": 403, "errorMessage": error, "data": []})

            elif input_file == "WRONG_EXTS":
                os.remove(os.path.join(UPLOAD_FOLDER, input_file))
                error = "This file is not supported"
                return JSONResponse(status_code=404, content={"errorCode": 404, "errorMessage": error, "data": []})

        """TODO: Preprocessing to crop image"""

        return await extract_info()

    except Exception as e:
        return JSONResponse(status_code=501, content={"errorCode": 501, "errorMessage": e, "data": []})


@app.post("/extract")
async def extract_info():
    global OFFSET, NMS_THRESHOLD

    # Check if uploaded image exists
    input_images_list = os.listdir(UPLOAD_FOLDER)
    if input_images_list is not None and len(input_images_list) >= 2:
        front_image = os.path.join(UPLOAD_FOLDER, input_images_list[0])
        back_image = os.path.join(UPLOAD_FOLDER, input_images_list[1])

    """FRONT IMAGE"""
    # Detect corner
    corner_model = CORNER_MODEL(front_image)
    predictions = corner_model.pred[0]
    categories = predictions[:, 5].tolist()  # get class
    if len(categories) != 4:
        error = "Detecting corner failed"
        return JSONResponse(status_code=401, content={"errorCode": 401, "errorMessage": error, "data": []})

    # get coordinates of corner boxes(x1, x2, y1, y2)
    boxes = utils.class_order(predictions[:, :4].tolist(), categories)

    image = Image.open(front_image)
    center_points = list(map(utils.get_center_point, boxes))

    """TODO: Temporary fixing"""
    c2, c3 = center_points[2], center_points[3]
    c2_fix, c3_fix = (c2[0], c2[1] + OFFSET), (c3[0], c3[1] + OFFSET)

    center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
    center_points = np.asarray(center_points)

    aligned = utils.four_point_transform(image, center_points)
    aligned = Image.fromarray(aligned)

    content_model = CONTENT_MODEL(aligned)
    predictions = content_model.pred[0]
    categories = predictions[:, 5].tolist()  # get class

    if 7 not in categories and len(categories) < 9:
        error = "Missing fields! Detecting content failed!"
        return JSONResponse(status_code=402, content={"errorCode": 402, "errorMessage": error, "data": []})

    elif 7 in categories and len(categories) < 10:
        error = "Missing fields! Detecting content failed!"
        return JSONResponse(status_code=402, content={"errorCode": 402, "errorMessage": error, "data": []})

    boxes = predictions[:, :4].tolist()

    """Non Maximum Suppression"""
    boxes, categories = utils.non_maximum_suppression(np.array(boxes), categories, NMS_THRESHOLD)

    boxes = utils.class_order(boxes, categories)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    else:
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))

    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if 5 < index < 9:
            right = right + 100
        cropped_image = aligned.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(SAVE_DIR, f'{index}.jpg'))

    detected_fields = []  # Collecting all detected parts
    for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
        if idx > 0:
            img_ = Image.open(os.path.join(SAVE_DIR, img_crop))
            s = detector.predict(img_)
            detected_fields.append(s)

    if 7 in categories:
        detected_fields = detected_fields[:6] + [detected_fields[6] + ', ' + detected_fields[7]] + [detected_fields[8]]

    face_img_path = os.path.join(SAVE_DIR, f'0.jpg')

    if rekognition.check_existed_face(face_img_path) != None:
        print("EXISTED FACE!")
    else:
        face_id = rekognition.add_face_to_collection(face_img_path)
        database_management.add_record_to_db(detected_fields, face_id, 'clients')

    """BACK IMAGE"""
    back_image_data = cv2.imread(back_image)
    d = pytesseract.image_to_data(back_image_data, output_type=Output.DICT)

    date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

    n_boxes = len(d['text'])
    date_of_issue = None
    for i in range(n_boxes):
        if int(d['conf'][i]) > 80:
            if re.match(date_pattern, d['text'][i]):
                date_of_issue = d['text'][i] 

    if date_of_issue == None:
        error = "Missing fields! Cannot detect date of issue!"
        return JSONResponse(status_code=402, content={"errorCode": 402, "errorMessage": error, "data": []})

    response = {
        "errorCode": 0,
        "errorMessage": '',
        "data": [
            {
                "No.": detected_fields[0],
                "fullName": detected_fields[1],
                "dob": detected_fields[2],
                "sex": detected_fields[3],
                "nationality": detected_fields[4],
                "dateOfExpiry": detected_fields[7]
            },
            {
                'placeOfIssue': 'Cục Cảnh sát Quản lý hành chính về trật tự xã hội',
                'dateOfIssue': date_of_issue
            }
        ]
    }

    response = jsonable_encoder(response)

    return JSONResponse(content=response)


@app.post("/download")
async def download(file: str = Form(...)):
    if file != 'undefined':
        noti = 'Download file successfully!'
        return JSONResponse(status_code=201, content={"message": noti})
    else:
        error = 'No file to download!'
        return JSONResponse(status_code=405, content={"message": error})
