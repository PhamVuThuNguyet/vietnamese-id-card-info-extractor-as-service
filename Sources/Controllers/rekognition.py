import Sources.Controllers.config as cfg
from Sources.Controllers import boto3_endpoint

# Amazon Rekognition config
COLLECTION_ID = cfg.COLLECTION_ID
client = boto3_endpoint.create_boto_client('rekognition')


def check_existed_face(image_path):
    global client

    with open(image_path, 'rb') as image_file:
        search_image = image_file.read()

    # Search for the face in the collection
    response = client.search_faces_by_image(
        CollectionId=COLLECTION_ID,
        FaceMatchThreshold=0.95,
        Image={'Bytes': search_image}
    )

    # Check the search results
    if len(response['FaceMatches']) > 0:
        return response['FaceMatches'][0]['Face']['FaceId']
    else:
        return None


def add_face_to_collection(image_path):
    global client

    with open(image_path, 'rb') as image_file:
        search_image = image_file.read()

    # Search for the face in the collection
    response = client.index_faces(
        CollectionId=COLLECTION_ID,
        Image={'Bytes': search_image}
    )
    return response["FaceRecords"][0]["Face"]["FaceId"]


def delete_face_from_collection(face_id):
    global client

    response = client.delete_faces(
        CollectionId=COLLECTION_ID,
        FaceIds=[face_id]
    )

    if len(response["DeletedFaces"]) > 0:
        return True
    else:
        return False
