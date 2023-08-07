import os
import re
import pytesseract
from pytesseract import Output
import cv2
import Sources.Controllers.config as cfg
from Sources import app
from Sources.Controllers import utils
from fastapi import UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import yolov5
import numpy as np
from PIL import Image, ImageOps


""" ---- Setup ---- """
# Constant
OFFSET = 50

# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER_BACK
SAVE_DIR = cfg.SAVE_DIR_BACK

@app.post("/uploadBack")
async def upload(backFile: UploadFile = File(...)):
    try:
        if not os.path.isdir(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        input_images_list = os.listdir(UPLOAD_FOLDER)
        if input_images_list is not None:
            for uploaded_image in input_images_list:
                os.remove(os.path.join(UPLOAD_FOLDER, uploaded_image))

        back_file_location = f'{UPLOAD_FOLDER}/backFile.jpg'

        """BACK FILE"""
        back_data = await backFile.read()
        with open(back_file_location, 'wb') as f:
            f.write(back_data)

        # Validating file
        input_file = os.listdir(UPLOAD_FOLDER)[0]
        
        if input_file == 'NULL':
            os.remove(os.path.join(UPLOAD_FOLDER, input_file))
            error = "No file selected!"
            return JSONResponse(status_code=400, content={"errorCode": 400, "errorMessage": error, "data": []})

        elif input_file == "WRONG_EXTS":
            os.remove(os.path.join(UPLOAD_FOLDER, input_file))
            error = "This file is not supported"
            return JSONResponse(status_code=400, content={"errorCode": 400, "errorMessage": error, "data": []})

        """TODO: Preprocessing to crop image"""

        return await extract_info()

    except Exception as e:
        return JSONResponse(status_code=500, content={"errorCode": 500, "errorMessage": str(e), "data": []})
    
@app.post("/extractBack")
async def extract_info():
    global OFFSET

    # Check if uploaded image exists
    input_images_list = os.listdir(UPLOAD_FOLDER)
    if input_images_list is not None:
        back_image = f'{UPLOAD_FOLDER}/backFile.jpg'

    """BACK IMAGE"""
    # Detect corner
    corner_model = CORNER_MODEL(back_image)
    predictions = corner_model.pred[0]
    categories = predictions[:, 5].tolist()  # get class
    if len(categories) != 4:
        error = "Detecting corner failed"
        return JSONResponse(status_code=500, content={"errorCode": 500, "errorMessage": error, "data": []})

    # get coordinates of corner boxes(x1, x2, y1, y2)
    boxes = utils.class_order(predictions[:, :4].tolist(), categories)

    center_points = list(map(utils.get_center_point, boxes))

    """TODO: Temporary fixing"""
    c2, c3 = center_points[2], center_points[3]
    c2_fix, c3_fix = (c2[0], c2[1] + OFFSET), (c3[0], c3[1] + OFFSET)

    c0, c1 = center_points[0], center_points[1]
    c0_fix, c1_fix = (c0[0], c0[1] - OFFSET), (c1[0], c1[1] - OFFSET)

    center_points = [c0_fix, c1_fix, c2_fix, c3_fix]
    center_points = np.asarray(center_points)
    
    image = Image.open(back_image)
    image = ImageOps.exif_transpose(image)

    aligned = utils.four_point_transform(image, center_points)
    aligned = Image.fromarray(aligned)

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))

    aligned.save(os.path.join(SAVE_DIR, f'newBackFile.jpg'), dpi=(300, 300))

    back_image_data = cv2.imread(os.path.join(SAVE_DIR, f'newBackFile.jpg'), cv2.IMREAD_GRAYSCALE)
    back_image_data = cv2.resize(back_image_data, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    # kernel = np.ones((1, 1), np.uint8)  
    # back_image_data = cv2.dilate(back_image_data, kernel, iterations=1) 
    # back_image_data = cv2.erode(back_image_data, kernel, iterations=1)  
    
    # back_image_data = cv2.GaussianBlur(back_image_data, (5, 5), 5)
    # back_image_data = cv2.threshold(back_image_data, 0, 125, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # cv2.imwrite(os.path.join(SAVE_DIR, f'newBackFile.jpg'), back_image_data)

    d = pytesseract.image_to_data(back_image_data, output_type=Output.DICT, config= '--oem 3 --psm 6')

    date_pattern = r'\d{1,2}/\d{1,2}/\d{4}\b'
    check_pattern = 'CANH'

    n_boxes = len(d['text'])
    date_of_issue = None
    check = False
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            print(d['text'][i])
            date_of_issue_find = re.search(date_pattern, d['text'][i])
            if date_of_issue_find:
                date_of_issue = date_of_issue_find.group()  
            if re.search(check_pattern, d['text'][i]):
                check = True

    if date_of_issue == None:
        error = "Missing fields! Cannot detect date of issue!"
        return JSONResponse(status_code=500, content={"errorCode": 500, "errorMessage": error, "data": []})

    if not check:
        error = "Wrong image! Please capture the back side of your identity card!"
        return JSONResponse(status_code=500, content={"errorCode": 500, "errorMessage": error, "data": []})
    
    response = {
        "errorCode": 0,
        "errorMessage": '',
        "data": [
            {
                'placeOfIssue': 'Cục Cảnh sát Quản lý hành chính về trật tự xã hội',
                'dateOfIssue': date_of_issue
            }
        ]
    }

    response = jsonable_encoder(response)

    return JSONResponse(content=response)
