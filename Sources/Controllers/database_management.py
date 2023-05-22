import logging

from Sources.Controllers import boto3_endpoint
from botocore.exceptions import ClientError

client = boto3_endpoint.create_boto_client('dynamodb')


def create_table(table_name):
    global client

    try:
        response = client.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'face_id', 'KeyType': 'HASH'},  # Partition key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'face_id', 'AttributeType': 'S'},
            ],
            ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10})
        print(response)
    except ClientError as err:
        logging.error(
            "Couldn't create table %s. Here's why: %s: %s", table_name,
            err.response['Error']['Code'], err.response['Error']['Message'])
        raise


def delete_table(table_name):
    global client

    response = client.delete_table(
        TableName=table_name
    )

    print(response)


def add_record_to_db(detected_fields, face_id, table_name):
    global client

    sex = '1' if detected_fields[3] == "Nữ" else '0'

    response = client.put_item(
        TableName=table_name,
        Item={
            'face_id': {
                'S': face_id
            },
            'name': {
                'S': detected_fields[1]
            },
            'dob': {
                'S': detected_fields[2]
            },
            'sex': {
                'N': sex
            },
            'nationality': {
                'S': detected_fields[4]
            },
            'poo': {
                'S': detected_fields[5]
            },
            'por': {
                'S': detected_fields[6]
            },
            'doe': {
                'S': detected_fields[7]
            }
        }
    )

    print(response)

# if __name__ == "__main__":
#     create_table('clients')
