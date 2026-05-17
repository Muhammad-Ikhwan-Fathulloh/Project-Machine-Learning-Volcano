import os
import uuid
import requests
import json
from datetime import datetime
import boto3
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# AWS / LocalStack Connection Details
USE_LOCALSTACK = os.getenv("USE_LOCALSTACK", "True").lower() == "true"
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test")

# S3 & DynamoDB Target Resources
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "volcano-classifier-models")
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "Volcano_Dataset")
TOPIC_NAME = os.getenv("SNS_TOPIC_NAME", "Volcano_Alerts")
QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "Volcano_Inference_Queue")

# Telegram Bot Integration (Optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_aws_client(service_name):
    """Factory to get the correct boto3 client based on LocalStack or Real AWS mode"""
    kwargs = {
        'region_name': AWS_REGION,
        'aws_access_key_id': AWS_ACCESS_KEY,
        'aws_secret_access_key': AWS_SECRET_KEY
    }
    if USE_LOCALSTACK and LOCALSTACK_ENDPOINT:
        kwargs['endpoint_url'] = LOCALSTACK_ENDPOINT
    return boto3.client(service_name, **kwargs)

def get_aws_resource(service_name):
    """Factory to get the correct boto3 resource based on LocalStack or Real AWS mode"""
    kwargs = {
        'region_name': AWS_REGION,
        'aws_access_key_id': AWS_ACCESS_KEY,
        'aws_secret_access_key': AWS_SECRET_KEY
    }
    if USE_LOCALSTACK and LOCALSTACK_ENDPOINT:
        kwargs['endpoint_url'] = LOCALSTACK_ENDPOINT
    return boto3.resource(service_name, **kwargs)

# Initialize AWS clients and resources
s3 = get_aws_client('s3')
dynamodb = get_aws_resource('dynamodb')
sns = get_aws_client('sns')
sqs = get_aws_client('sqs')
cloudwatch = get_aws_client('cloudwatch')

def init_resources():
    """Create all AWS resources (S3, DynamoDB, SQS, SNS) if they do not exist"""
    try:
        # 1. DynamoDB Table Initialization
        existing_tables = [t.name for t in dynamodb.tables.all()]
        if TABLE_NAME not in existing_tables:
            print(f"Creating DynamoDB table: {TABLE_NAME}...")
            dynamodb.create_table(
                TableName=TABLE_NAME,
                KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            )
            print(f"✅ DynamoDB Table '{TABLE_NAME}' created.")
        else:
            print(f"ℹ️ DynamoDB Table '{TABLE_NAME}' already exists.")

        # 2. S3 Bucket Initialization
        buckets = s3.list_buckets().get('Buckets', [])
        if not any(b['Name'] == BUCKET_NAME for b in buckets):
            print(f"Creating S3 Bucket: {BUCKET_NAME}...")
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=BUCKET_NAME)
            else:
                s3.create_bucket(
                    Bucket=BUCKET_NAME,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )
            print(f"✅ S3 Bucket '{BUCKET_NAME}' created.")
        else:
            print(f"ℹ️ S3 Bucket '{BUCKET_NAME}' already exists.")

        # 3. SNS Topic Initialization
        sns.create_topic(Name=TOPIC_NAME)
        print(f"✅ SNS Topic '{TOPIC_NAME}' ensured.")

        # 4. SQS Queue Initialization
        sqs.create_queue(QueueName=QUEUE_NAME)
        print(f"✅ SQS Queue '{QUEUE_NAME}' ensured.")

    except Exception as e:
        print(f"❌ AWS Initialization Error: {e}")

# S3 Helpers
def upload_model_to_s3(file_path, object_name):
    """Upload trained local model/label encoder files to S3 bucket"""
    try:
        s3.upload_file(file_path, BUCKET_NAME, object_name)
        print(f"✅ Successfully uploaded '{object_name}' to S3.")
        return True
    except Exception as e:
        print(f"❌ S3 Upload Error for '{object_name}': {e}")
        return False

def download_model_from_s3(file_path, object_name):
    """Download trained model/label encoder files from S3 bucket"""
    try:
        s3.download_file(BUCKET_NAME, object_name, file_path)
        print(f"✅ Successfully downloaded '{object_name}' from S3.")
        return True
    except Exception as e:
        print(f"ℹ️ Could not download '{object_name}' from S3 (might not exist yet): {e}")
        return False

# DynamoDB Operations
def log_inference_to_dynamodb(id_str, tinggi_meter, lat, lon, prediction, confidence):
    """Log prediction inputs and output to the unified Volcano_Dataset DynamoDB Table"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        item = {
            'id': id_str,
            'tinggi_meter': float(tinggi_meter),
            'lat': float(lat),
            'lon': float(lon),
            'predicted_bentuk': prediction,
            'confidence': float(confidence),
            'is_training_sample': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        table.put_item(Item=item)
        print(f"✅ Inference logged to DynamoDB: {id_str}")
        return True
    except Exception as e:
        print(f"❌ DynamoDB Log Inference Error: {e}")
        return False

def add_labeled_sample_to_dynamodb(tinggi_meter, lat, lon, bentuk):
    """Add a verified training coordinate and shape label directly to DynamoDB"""
    try:
        id_str = str(uuid.uuid4())
        table = dynamodb.Table(TABLE_NAME)
        item = {
            'id': id_str,
            'tinggi_meter': float(tinggi_meter),
            'lat': float(lat),
            'lon': float(lon),
            'bentuk': bentuk,
            'is_training_sample': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        table.put_item(Item=item)
        print(f"✅ Labeled training sample added to DynamoDB: {id_str}")
        return id_str
    except Exception as e:
        print(f"❌ DynamoDB Add Labeled Sample Error: {e}")
        return None

def verify_prediction_log(id_str, actual_bentuk):
    """Verify a previous prediction log and convert it to a training sample in DynamoDB"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.get_item(Key={'id': id_str})
        item = response.get('Item')
        if not item:
            print(f"⚠️ Item with ID '{id_str}' not found in DynamoDB.")
            return False

        table.update_item(
            Key={'id': id_str},
            UpdateExpression="SET is_training_sample = :t, bentuk = :b, actual_bentuk = :b",
            ExpressionAttributeValues={
                ':t': True,
                ':b': actual_bentuk
            }
        )
        print(f"✅ Prediction log '{id_str}' verified and promoted to training sample.")
        return True
    except Exception as e:
        print(f"❌ DynamoDB Verify Log Error: {e}")
        return False

def get_training_samples_from_dynamodb():
    """Retrieve all custom training samples from DynamoDB"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.scan(
            FilterExpression="is_training_sample = :t",
            ExpressionAttributeValues={":t": True}
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ DynamoDB Scan Error: {e}")
        return []

def get_all_logs_from_dynamodb():
    """Retrieve all records (logs + custom training data) from DynamoDB"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.scan()
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ DynamoDB Scan All Error: {e}")
        return []

# Notification and Queue Operations
def send_alert(message):
    """Publish a warning message to the SNS alert topic and optional Telegram channel"""
    try:
        topic_arn = sns.create_topic(Name=TOPIC_NAME)['TopicArn']
        sns.publish(TopicArn=topic_arn, Message=message, Subject="🌋 Volcano Classifier Warning")
        print(f"✅ Published SNS Alert: {message}")
        
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_alert(message)
        return True
    except Exception as e:
        print(f"❌ SNS Alert Error: {e}")
        return False

def send_telegram_alert(message):
    """Dispatch an alert message to a custom Telegram channel"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🚨 *VOLCANO AI ALERT* 🚨\n\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
        print("✅ Telegram alert sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Telegram Warning Dispatch Error: {e}")
        return False

def push_to_queue(message_body):
    """Push prediction results to SQS for async downstream operations"""
    try:
        queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)['QueueUrl']
        sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
        print(f"✅ Log pushed to SQS Queue: {QUEUE_NAME}")
        return True
    except Exception as e:
        print(f"❌ SQS Queue Error: {e}")
        return False

def log_metric(metric_name, value, unit='None'):
    """Record metrics to AWS CloudWatch for real-time monitoring"""
    try:
        cloudwatch.put_metric_data(
            Namespace='Volcano/Classifier',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': float(value),
                    'Unit': unit
                }
            ]
        )
        return True
    except Exception as e:
        print(f"❌ CloudWatch Put Metric Error: {e}")
        return False
