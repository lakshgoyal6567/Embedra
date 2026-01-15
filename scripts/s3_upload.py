import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv() # Load variables from .env file

# --- CONFIGURATION ---
BUCKET_NAME = "project_name-cloud-vault-v1" # Changed to be more unique
REGION = "us-east-1"

def create_bucket(s3_client, bucket_name):
    """Creates the S3 bucket if it doesn't exist."""
    print(f"Checking bucket '{bucket_name}'...")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' exists and is accessible.")
        return True
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            print(f"Bucket '{bucket_name}' not found. Creating it...")
            try:
                if REGION == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': REGION}
                    )
                print(f"✅ Bucket '{bucket_name}' created successfully.")
                return True
            except Exception as create_error:
                print(f"❌ Failed to create bucket: {create_error}")
                return False
        elif error_code == 403:
            print(f"❌ Access Denied: You do not have permission to access bucket '{bucket_name}'.")
            return False
        else:
            print(f"❌ Error checking bucket: {e}")
            return False

def upload_images_to_s3(local_dir):
    """
    Uploads all images in a directory to S3.
    """
    # Initialize S3 Client
    try:
        s3 = boto3.client('s3', region_name=REGION)
    except Exception as e:
        print(f"Failed to initialize S3 client: {e}")
        return

    # Ensure bucket exists
    if not create_bucket(s3, BUCKET_NAME):
        return

    print(f"Scanning '{local_dir}' for images...")
    
    files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.avif', '.webp'))]
    print(f"Found {len(files)} images.")

    print(f"Uploading to s3://{BUCKET_NAME}...")
    
    success_count = 0
    
    for file_name in files:
        local_path = os.path.join(local_dir, file_name)
        s3_key = f"raw_images/{file_name}" # Store in a folder
        
        try:
            # Upload
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            print(f"✅ Uploaded: {file_name}")
            success_count += 1
        except FileNotFoundError:
            print(f"❌ File not found: {local_path}")
        except NoCredentialsError:
            print("❌ AWS Credentials not available.")
            return
        except Exception as e:
            print(f"❌ Error uploading {file_name}: {e}")

    print(f"\nUpload Complete. {success_count}/{len(files)} images secured in the Cloud Vault.")

if __name__ == "__main__":
    # You need to run 'aws configure' or set AWS_ACCESS_KEY_ID env vars first.
    if not os.path.exists("data/raw_data"):
        print("Error: 'data/raw_data' folder not found.")
    else:
        upload_images_to_s3("data/raw_data")
