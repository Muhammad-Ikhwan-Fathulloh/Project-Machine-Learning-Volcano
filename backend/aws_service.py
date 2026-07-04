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

# Check if AWS is actually configured (either LocalStack or real AWS)
def is_aws_configured():
    """Check if AWS/LocalStack is properly configured"""
    try:
        # Try to get a client and make a simple call
        test_client = get_aws_client('s3')
        test_client.list_buckets()
        return True
    except Exception as e:
        print(f"⚠️ AWS/LocalStack not available: {e}")
        return False

AWS_ENABLED = is_aws_configured()

def get_aws_client(service_name):
    """Factory to get the correct boto3 client based on LocalStack or Real AWS mode"""
    if not AWS_ENABLED:
        # Return a dummy client that logs operations
        return DummyAWSClient(service_name)
    
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
    if not AWS_ENABLED:
        return DummyAWSResource(service_name)
    
    kwargs = {
        'region_name': AWS_REGION,
        'aws_access_key_id': AWS_ACCESS_KEY,
        'aws_secret_access_key': AWS_SECRET_KEY
    }
    if USE_LOCALSTACK and LOCALSTACK_ENDPOINT:
        kwargs['endpoint_url'] = LOCALSTACK_ENDPOINT
    return boto3.resource(service_name, **kwargs)

# ============================================
# DUMMY AWS CLIENTS (When AWS is not available)
# ============================================

class DummyAWSClient:
    """Dummy AWS client for local mode without AWS"""
    def __init__(self, service_name):
        self.service_name = service_name
        self.logs = []
        print(f"ℹ️ Running in LOCAL mode - {service_name} operations will be simulated")
    
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            # Log the operation
            print(f"📝 LOCAL MODE: {self.service_name}.{name} called with {args}, {kwargs}")
            # Return dummy responses
            if name == 'list_buckets':
                return {'Buckets': []}
            elif name == 'create_bucket':
                return {'Location': '/'}
            elif name == 'upload_file':
                return True
            elif name == 'download_file':
                return True
            elif name == 'head_bucket':
                return {}
            elif name == 'list_queues':
                return {'QueueUrls': []}
            elif name == 'create_queue':
                return {'QueueUrl': f'http://localhost/{self.service_name}'}
            elif name == 'get_queue_url':
                return {'QueueUrl': f'http://localhost/{self.service_name}'}
            elif name == 'send_message':
                return {'MessageId': str(uuid.uuid4())}
            elif name == 'publish':
                return {'MessageId': str(uuid.uuid4())}
            elif name == 'create_topic':
                return {'TopicArn': f'arn:aws:sns:local:{TOPIC_NAME}'}
            elif name == 'put_metric_data':
                return {}
            else:
                return {}
        return dummy_method

class DummyAWSResource:
    """Dummy AWS resource for local mode without AWS"""
    def __init__(self, service_name):
        self.service_name = service_name
        self.tables = []
        print(f"ℹ️ Running in LOCAL mode - {service_name} resources will be simulated")
    
    def __getattr__(self, name):
        if name == 'Table':
            class DummyTable:
                def __init__(self, table_name):
                    self.table_name = table_name
                    self.items = []
                
                def put_item(self, Item):
                    self.items.append(Item)
                    print(f"📝 LOCAL MODE: Saved item to {self.table_name}")
                    return {}
                
                def get_item(self, Key):
                    for item in self.items:
                        if item.get('id') == Key.get('id'):
                            return {'Item': item}
                    return {}
                
                def update_item(self, Key, UpdateExpression, ExpressionAttributeValues):
                    for item in self.items:
                        if item.get('id') == Key.get('id'):
                            if 'is_training_sample' in ExpressionAttributeValues:
                                item['is_training_sample'] = ExpressionAttributeValues[':t']
                            if 'bentuk' in ExpressionAttributeValues:
                                item['bentuk'] = ExpressionAttributeValues[':b']
                            return {}
                    return {}
                
                def scan(self, FilterExpression=None, ExpressionAttributeValues=None):
                    if FilterExpression and ExpressionAttributeValues:
                        filtered = [item for item in self.items if item.get('is_training_sample') == ExpressionAttributeValues.get(':t')]
                        return {'Items': filtered}
                    return {'Items': self.items}
                
                @property
                def table_status(self):
                    return 'ACTIVE'
            
            return DummyTable
        
        elif name == 'tables':
            class DummyTableCollection:
                def all(self):
                    return []
            return DummyTableCollection()
        else:
            return lambda *args, **kwargs: {}

# Initialize AWS clients and resources
s3 = get_aws_client('s3')
dynamodb = get_aws_resource('dynamodb')
sns = get_aws_client('sns')
sqs = get_aws_client('sqs')
cloudwatch = get_aws_client('cloudwatch')

def init_resources():
    """Create all AWS resources (S3, DynamoDB, SQS, SNS) if they do not exist"""
    if not AWS_ENABLED:
        print("📁 Running in LOCAL mode - using simulated resources")
        print("   All data will be stored in memory")
        return
    
    try:
        # 1. DynamoDB Table Initialization
        try:
            existing_tables = [t.name for t in dynamodb.tables.all()]
        except:
            existing_tables = []
        
        if TABLE_NAME not in existing_tables:
            print(f"Creating DynamoDB table: {TABLE_NAME}...")
            try:
                dynamodb.create_table(
                    TableName=TABLE_NAME,
                    KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
                    ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                )
                print(f"✅ DynamoDB Table '{TABLE_NAME}' created.")
            except Exception as e:
                print(f"⚠️ Could not create DynamoDB table: {e}")
        else:
            print(f"ℹ️ DynamoDB Table '{TABLE_NAME}' already exists.")

        # 2. S3 Bucket Initialization
        try:
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
        except Exception as e:
            print(f"⚠️ S3 bucket initialization warning: {e}")

        # 3. SNS Topic Initialization
        try:
            sns.create_topic(Name=TOPIC_NAME)
            print(f"✅ SNS Topic '{TOPIC_NAME}' ensured.")
        except Exception as e:
            print(f"⚠️ SNS topic initialization warning: {e}")

        # 4. SQS Queue Initialization
        try:
            sqs.create_queue(QueueName=QUEUE_NAME)
            print(f"✅ SQS Queue '{QUEUE_NAME}' ensured.")
        except Exception as e:
            print(f"⚠️ SQS queue initialization warning: {e}")

    except Exception as e:
        print(f"❌ AWS Initialization Error: {e}")

# S3 Helpers
def upload_model_to_s3(file_path, object_name):
    """Upload trained local model/label encoder files to S3 bucket"""
    if not AWS_ENABLED:
        print(f"📝 LOCAL MODE: Would upload {object_name} to S3")
        return True
    
    try:
        s3.upload_file(file_path, BUCKET_NAME, object_name)
        print(f"✅ Successfully uploaded '{object_name}' to S3.")
        return True
    except Exception as e:
        print(f"❌ S3 Upload Error for '{object_name}': {e}")
        return False

def download_model_from_s3(file_path, object_name):
    """Download trained model/label encoder files from S3 bucket"""
    if not AWS_ENABLED:
        print(f"📝 LOCAL MODE: Would download {object_name} from S3")
        return False
    
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
        # Fallback: log to file
        try:
            log_file = 'local_inference_logs.json'
            existing = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            existing.append(item)
            with open(log_file, 'w') as f:
                json.dump(existing, f, indent=2, default=str)
            print(f"📝 Inference logged to local file: {log_file}")
        except:
            pass
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
        # Fallback: log to file
        try:
            log_file = 'local_training_samples.json'
            existing = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            existing.append(item)
            with open(log_file, 'w') as f:
                json.dump(existing, f, indent=2, default=str)
            print(f"📝 Training sample saved to local file: {log_file}")
        except:
            pass
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
    if not AWS_ENABLED:
        # Try to read from local file
        try:
            log_file = 'local_training_samples.json'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
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
    if not AWS_ENABLED:
        all_logs = []
        # Try to read from local files
        try:
            log_file = 'local_inference_logs.json'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    all_logs.extend(json.load(f))
        except:
            pass
        try:
            training_file = 'local_training_samples.json'
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    all_logs.extend(json.load(f))
        except:
            pass
        return all_logs
    
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.scan()
        items = response.get('Items', [])
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        
        return items
    except Exception as e:
        print(f"❌ DynamoDB Scan All Error: {e}")
        return []

# Notification and Queue Operations
def send_alert(message):
    """Publish a warning message to the SNS alert topic and optional Telegram channel"""
    print(f"⚠️ ALERT: {message}")
    
    if not AWS_ENABLED:
        # Log alert to file
        try:
            alert_file = 'local_alerts.json'
            existing = []
            if os.path.exists(alert_file):
                with open(alert_file, 'r') as f:
                    existing = json.load(f)
            existing.append({
                'timestamp': datetime.utcnow().isoformat(),
                'message': message
            })
            with open(alert_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except:
            pass
        return True
    
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
    if not AWS_ENABLED:
        # Log to file
        try:
            queue_file = 'local_queue_messages.json'
            existing = []
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    existing = json.load(f)
            existing.append({
                'timestamp': datetime.utcnow().isoformat(),
                'message': message_body
            })
            with open(queue_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except:
            pass
        return True
    
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
    if not AWS_ENABLED:
        # Log to file
        try:
            metric_file = 'local_metrics.json'
            existing = []
            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    existing = json.load(f)
            existing.append({
                'timestamp': datetime.utcnow().isoformat(),
                'metric': metric_name,
                'value': float(value),
                'unit': unit
            })
            with open(metric_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except:
            pass
        return True
    
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