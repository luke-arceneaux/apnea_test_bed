''' 
S3 client for fetching/uploading files to the cloud.

Ensure that boto3 is installed (pip install boto3)
For more examples see this link: 
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-examples.html

'''
# from distutils.command.upload import upload
import boto3
from botocore.exceptions import ClientError
import logging
import pandas as pd
import os
import io
import warnings

# Prefer environment variables; optional legacy module-level overrides.
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

if not ACCESS_KEY or not SECRET_KEY:
    warnings.warn(
        "AWS credentials not set. Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
        "before using S3Client.",
        stacklevel=2,
    )

''' Instantiate S3 client'''
class S3Client:
    def __init__(self, ACCESS_KEY=ACCESS_KEY,\
                       SECRET_KEY=SECRET_KEY,\
                       BUCKET_NAME=''):

        self.ACCESS_KEY = ACCESS_KEY
        self.SECRET_KEY = SECRET_KEY
        self.bucket_name=BUCKET_NAME

                
        # Creating the low level functional client
        region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        self.client = boto3.client(
            's3',
            aws_access_key_id = self.ACCESS_KEY,
            aws_secret_access_key = self.SECRET_KEY,
            region_name = region,
        )
            
        # Creating the high level object oriented interface
        self.resource = boto3.resource(
            's3',
            aws_access_key_id = self.ACCESS_KEY,
            aws_secret_access_key = self.SECRET_KEY,
            region_name = region,
        )
        

    # List buckets associated with AWS account
    def list_buckets(self):
        # Fetch the list of existing buckets and print
        clientResponse = self.client.list_buckets()
        print('Printing bucket names...')
        for bucket in clientResponse['Buckets']:
            print(f'Bucket Name: {bucket["Name"]}')

    # Upload df as csv to S3 bucket
    def upload_df(self, df, df_file, bucket_name=''):
        if not bucket_name:
            bucket_name = self.bucket_name
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, sep=',', header=True, index=False)#False)
        self.resource.Object(bucket_name, df_file).put(Body=csv_buffer.getvalue())
        print(f'[S3 INFO]: Uploaded df {df_file} to bucket {bucket_name}!')

    # Upload a file to S3 bucket
    def upload_file(self, file_name, upload_file_name=None, bucket_name='', content_type=None):

        if not bucket_name:
            bucket_name = self.bucket_name

        # If no upload_file_name specified, use the same filename as the file being uploaded
        if upload_file_name is None:
            upload_file_name = file_name
            
        extra_args = {'ContentType': content_type} if content_type else {}

        # Upload the file
        try:
            response = self.client.upload_file(file_name, bucket_name, upload_file_name, ExtraArgs=extra_args)
        except ClientError as e:
            print(f'Error uploading file {file_name}: {e}')
            logging.error(e)
            return False

        print(f'[S3 INFO]: Uploaded file {file_name} to bucket {bucket_name}!')
        return True
    
    # Download a wav file from S3
    def get_file_wav(self, file_name, temp_dir='temp/', bucket_name=''):

        if not bucket_name:
            bucket_name = self.bucket_name 
        out_path = f'{temp_dir}{file_name}'
        print(f'Getting wav file {out_path}......')
        base_path = os.path.dirname(out_path)
        if not os.path.exists(base_path): os.makedirs(base_path)
        self.client.download_file(bucket_name, file_name, out_path)
        print(f'[S3 INFO]: Downloaded wav file from bucket {bucket_name} to {out_path}!')
        return out_path

    # Reads file from S3 as a Pandas dataframe
    def get_file_df(self, file_name, bucket_name=''):

        if not bucket_name:
            bucket_name = self.bucket_name 
        obj = self.client.get_object(Bucket=bucket_name, Key=file_name) # bucket, key
        df = pd.read_csv(io.BytesIO(obj['Body'].read()),  index_col=False)#on_bad_lines='skip', index_col=False)
        df.rename(columns=str.lower) # lowercase columns
        print(f'[S3 INFO]: Read file from {bucket_name} as df {file_name}!')
        return df, file_name

    # List files in S3 with prefix <prefix>
    def list_files(self, prefix="expert_annot/"):
        bucket = self.resource.Bucket(self.bucket_name)
        return [obj.key for obj in bucket.objects.filter(Prefix=prefix)]

