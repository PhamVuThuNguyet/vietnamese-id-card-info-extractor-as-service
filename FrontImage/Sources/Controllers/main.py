import logging
import os
from datetime import datetime

import httpx
import numpy as np
import Sources.Controllers.config as cfg
import yolov5
from fastapi import File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from Sources import app
from Sources.Controllers import utils
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
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER_FRONT
SAVE_DIR = cfg.SAVE_DIR_FRONT
BACK_API = cfg.BACK_API

""" Recognizion detected parts in ID """
config = Cfg.load_config_from_name(
    "vgg_seq2seq"
)  # OR vgg_transformer -> acc || vgg_seq2seq -> time
config["weights"] = cfg.OCR_MODEL_PATH
config["cnn"]["pretrained"] = False
config["device"] = cfg.DEVICE
config["predictor"]["beamsearch"] = False
detector = Predictor(config)


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


@app.post("/uploadFront")
async def upload(frontFile: UploadFile = File(...)):
    try:
        if not os.path.isdir(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        input_images_list = os.listdir(UPLOAD_FOLDER)
        if input_images_list is not None:
            for uploaded_image in input_images_list:
                os.remove(os.path.join(UPLOAD_FOLDER, uploaded_image))

        front_file_location = f"{UPLOAD_FOLDER}/frontFile.jpg"

        """FRONT FILE"""
        front_data = await frontFile.read()
        with open(front_file_location, "wb") as f:
            f.write(front_data)

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
            content={"errorCode": 500, "errorMessage": str(e), "data": []},
        )


@app.post("/extractFront")
async def extract_info():
    try:
        global OFFSET, NMS_THRESHOLD

        # Check if uploaded image exists
        input_images_list = os.listdir(UPLOAD_FOLDER)
        if input_images_list is not None:
            front_image = f"{UPLOAD_FOLDER}/frontFile.jpg"

        with open(front_image, "rb") as f:
            files = {"backFile": (front_image, f)}
            timeout = httpx.Timeout(connect=60.0, read=60.0, write=60.0)
            with httpx.Client(timeout=timeout) as client:
                response = await client.post(BACK_API, files=files)

        if response.json()["errorCode"] == 0:
            data = response.json()["data"][0]
            print(data)
            issue_age = (
                int(data["doi"].split("/")[-1])
                - int(data["dob"].split("/")[-1])
                - (
                    (int(data["doi"].split("/")[1]), int(data["doi"].split("/")[0]))
                    < (int(data["dob"].split("/")[1]), int(data["dob"].split("/")[0]))
                )
            )
            if issue_age >= 14 and issue_age < 23:
                doe = (
                    str(data["dob"].split("/")[0])
                    + "/"
                    + str(data["dob"].split("/")[1])
                    + "/"
                    + str(int(data["dob"].split("/")[-1]) + 25)
                )
            elif issue_age >= 23 and issue_age < 38:
                doe = (
                    str(data["dob"].split("/")[0])
                    + "/"
                    + str(data["dob"].split("/")[1])
                    + "/"
                    + str(int(data["dob"].split("/")[-1]) + 40)
                )
            else:
                doe = (
                    str(data["dob"].split("/")[0])
                    + "/"
                    + str(data["dob"].split("/")[1])
                    + "/"
                    + str(int(data["dob"].split("/")[-1]) + 60)
                )
            response = {
                "errorCode": 0,
                "errorMessage": "",
                "data": [
                    {
                        "No.": data["No."],
                        "fullName": data["fullName"],
                        "dob": data["dob"],
                        "sex": data["sex"],
                        "nationality": "Việt Nam",
                        "dateOfExpiry": doe,
                    },
                    {
                        "placeOfIssue": "Cục Cảnh sát Quản lý hành chính về trật tự xã hội",
                        "dateOfIssue": data["doi"],
                    },
                ],
            }
            response = jsonable_encoder(response)

            return JSONResponse(content=response)
        else:
            """FRONT IMAGE"""
            # Detect corner
            front_image = f"{UPLOAD_FOLDER}/frontFile.jpg"
            corner_model = CORNER_MODEL(front_image)
            predictions = corner_model.pred[0]
            categories = predictions[:, 5].tolist()  # get class
            if len(categories) != 4:
                error = "Detecting corner failed"
                return JSONResponse(
                    status_code=500,
                    content={"errorCode": 500, "errorMessage": error, "data": []},
                )

            # get coordinates of corner boxes(x1, x2, y1, y2)
            boxes = utils.class_order(predictions[:, :4].tolist(), categories)

            center_points = list(map(utils.get_center_point, boxes))

            """TODO: Temporary fixing"""
            c2, c3 = center_points[2], center_points[3]
            c2_fix, c3_fix = (c2[0], c2[1] + OFFSET), (c3[0], c3[1] + OFFSET)

            center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
            center_points = np.asarray(center_points)

            image = Image.open(front_image)
            image = ImageOps.exif_transpose(image)

            aligned = utils.four_point_transform(image, center_points)
            aligned = Image.fromarray(aligned)
            aligned = aligned.convert("RGB")

            content_model = CONTENT_MODEL(aligned)
            predictions = content_model.pred[0]
            categories = predictions[:, 5].tolist()  # get class

            if 7 not in categories and len(categories) < 9:
                error = "Missing fields! Detecting content failed!"
                return JSONResponse(
                    status_code=500,
                    content={"errorCode": 500, "errorMessage": error, "data": []},
                )

            elif 7 in categories and len(categories) < 10:
                error = "Missing fields! Detecting content failed!"
                return JSONResponse(
                    status_code=500,
                    content={"errorCode": 500, "errorMessage": error, "data": []},
                )

            boxes = predictions[:, :4].tolist()

            """Non Maximum Suppression"""
            boxes, categories = utils.non_maximum_suppression(
                np.array(boxes), categories, NMS_THRESHOLD
            )

            boxes = utils.class_order(boxes, categories)
            if not os.path.isdir(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            else:
                for f in os.listdir(SAVE_DIR):
                    os.remove(os.path.join(SAVE_DIR, f))

            categories = sorted(categories)
            for index, box in enumerate(boxes):
                left, top, right, bottom = box
                if 5 < index < 9:
                    right = right + 100
                cropped_image = aligned.crop((left, top, right, bottom))
                cropped_image.save(
                    os.path.join(SAVE_DIR, f"{int(categories[index])}.jpg")
                )

            detected_fields = []  # Collecting all detected parts
            for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
                if idx > 0:
                    img_ = Image.open(os.path.join(SAVE_DIR, img_crop))
                    s = detector.predict(img_)
                    detected_fields.append(s)

            if 7 in categories:
                detected_fields = (
                    detected_fields[:6]
                    + [detected_fields[6] + ", " + detected_fields[7]]
                    + [detected_fields[8]]
                )

            # face_img_path = os.path.join(SAVE_DIR, f'0.jpg')

            # if rekognition.check_existed_face(face_img_path) != None:
            #     print("EXISTED FACE!")
            # else:
            #     face_id = rekognition.add_face_to_collection(face_img_path)
            #     database_management.add_record_to_db(detected_fields, face_id, 'clients')

            response = {
                "errorCode": 500,
                "errorMessage": "Cannot Detect Date Of Issue!",
                "data": [
                    {
                        "No.": detected_fields[0],
                        "fullName": detected_fields[1],
                        "dob": detected_fields[2],
                        "sex": detected_fields[3],
                        "nationality": detected_fields[4],
                        "dateOfExpiry": detected_fields[7],
                    },
                    {},
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
