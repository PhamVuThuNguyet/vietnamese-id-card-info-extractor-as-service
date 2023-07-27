import os
import re
import pytesseract
from pytesseract import Output
import cv2
import Sources.Controllers.config as cfg
from Sources import app
from fastapi import UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER_BACK

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
            return JSONResponse(status_code=403, content={"errorCode": 403, "errorMessage": error, "data": []})

        elif input_file == "WRONG_EXTS":
            os.remove(os.path.join(UPLOAD_FOLDER, input_file))
            error = "This file is not supported"
            return JSONResponse(status_code=404, content={"errorCode": 404, "errorMessage": error, "data": []})

        """TODO: Preprocessing to crop image"""

        return await extract_info()

    except Exception as e:
        return JSONResponse(status_code=501, content={"errorCode": 501, "errorMessage": str(e), "data": []})
    
@app.post("/extractBack")
async def extract_info():
    # Check if uploaded image exists
    input_images_list = os.listdir(UPLOAD_FOLDER)
    if input_images_list is not None:
        back_image = f'{UPLOAD_FOLDER}/backFile.jpg'

    """BACK IMAGE"""
    back_image_data = cv2.imread(back_image)
    d = pytesseract.image_to_data(back_image_data, output_type=Output.DICT, config= '--psm 6')

    date_pattern = '.*(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

    n_boxes = len(d['text'])
    date_of_issue = None
    for i in range(n_boxes):
        if int(d['conf'][i]) > 80:
            if re.search(date_pattern, d['text'][i]):
                date_of_issue = d['text'][i] 

    if date_of_issue == None:
        error = "Missing fields! Cannot detect date of issue!"
        return JSONResponse(status_code=402, content={"errorCode": 402, "errorMessage": error, "data": []})

    if (date_of_issue.find(":") != -1):
        date_of_issue = date_of_issue.split(":")[1]

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
