import logging
import os

import cv2
import Sources.Controllers.config as cfg
from fastapi import File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from qreader import QReader
from Sources import app

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER_BACK
SAVE_DIR = cfg.SAVE_DIR_BACK


@app.get("/")
async def kick_start():
    return JSONResponse(
        status_code=200,
        content={
            "errorCode": 0,
            "errorMessage": f"Start OK",
            "data": [],
        },
    )


@app.post("/uploadBack")
async def upload(backFile: UploadFile = File(...)):
    try:
        if not os.path.isdir(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        input_images_list = os.listdir(UPLOAD_FOLDER)
        if input_images_list is not None:
            for uploaded_image in input_images_list:
                os.remove(os.path.join(UPLOAD_FOLDER, uploaded_image))

        back_file_location = f"{UPLOAD_FOLDER}/backFile.jpg"

        """BACK FILE"""
        back_data = await backFile.read()
        with open(back_file_location, "wb") as f:
            f.write(back_data)

        # Validating file
        input_file = os.listdir(UPLOAD_FOLDER)[0]

        if input_file == "NULL":
            os.remove(os.path.join(UPLOAD_FOLDER, input_file))
            error = "No file selected!"
            return JSONResponse(
                status_code=400,
                content={"errorCode": 400, "errorMessage": error, "data": []},
            )

        elif input_file == "WRONG_EXTS":
            os.remove(os.path.join(UPLOAD_FOLDER, input_file))
            error = "This file is not supported"
            return JSONResponse(
                status_code=400,
                content={"errorCode": 400, "errorMessage": error, "data": []},
            )

        return await extract_info()

    except Exception as e:
        logging.exception(e)
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": 500,
                "errorMessage": f"Fail to upload! {str(e)}",
                "data": [],
            },
        )


@app.post("/extractBack")
async def extract_info():
    try:
        qreader = QReader(model_size="l", min_confidence=0.5, reencode_to="cp65001")
        image = cv2.cvtColor(
            cv2.imread(f"{UPLOAD_FOLDER}/backFile.jpg"), cv2.COLOR_BGR2RGB
        )

        # Use the detect_and_decode function to get the decoded QR data
        decoded_text = qreader.detect_and_decode(image=image)
        print(decoded_text)
        if decoded_text[0] is None:
            return JSONResponse(
                status_code=200,
                content={
                    "errorCode": 500,
                    "errorMessage": f"Fail to read QR Code!",
                    "data": [],
                },
            )
        data = decoded_text[0].split("|")
        number = data[0]
        fullname = data[2]
        dob = data[3]
        dob = dob[:2] + "/" + dob[2:4] + "/" + dob[4:]
        sex = data[4]
        address = data[5]
        date_of_issue = data[-1]
        date_of_issue = (
            date_of_issue[:2] + "/" + date_of_issue[2:4] + "/" + date_of_issue[4:]
        )

        response = {
            "errorCode": 0,
            "errorMessage": "",
            "data": [
                {
                    "No.": number,
                    "fullName": fullname,
                    "dob": dob,
                    "sex": sex,
                    "address": address,
                    "doi": date_of_issue,
                }
            ],
        }

        response = jsonable_encoder(response)

        return JSONResponse(content=response)
    except Exception as e:
        logging.exception(e)
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": 500,
                "errorMessage": f"Fail to read! {str(e)}",
                "data": [],
            },
        )
