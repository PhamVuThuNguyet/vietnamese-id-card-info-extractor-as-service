import boto3
from Sources.Controllers import config

ACCESS_KEY_ID = config.ACCESS_KEY_ID
SECRET_ACCESS_ID = config.SECRET_ACCESS_ID


def create_boto_client(service_name):
    return boto3.client(service_name, aws_access_key_id=ACCESS_KEY_ID,
                        aws_secret_access_key=SECRET_ACCESS_ID,
                        region_name="us-east-1")
