import cv2,boto3,botocore,gradio,random,os,json,io,string
import subprocess,time,shutil,base64,uuid,logging
import numpy as np
from collections import deque
from PIL import Image,ImageDraw
from datetime import datetime 
from botocore.exceptions import ClientError,NoCredentialsError, PartialCredentialsError

# Other Config
config = botocore.config.Config(
    read_timeout=900,
    connect_timeout=900,
    retries={"max_attempts": 0}
)

# Specified availability zones
region = 'us-east-1'

# AWS client initialization
session = boto3.Session()
s3_client = session.client('s3',region_name=region)
dynamodb = session.resource('dynamodb',region_name=region)
table = dynamodb.Table('videoInfo')
rekognition = session.client('rekognition',region_name=region)
bedrock = boto3.client('bedrock-runtime',region_name=region)
cloudwatchLog = boto3.client('logs',region_name=region)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region, config=config)
bedrock_agent = boto3.client("bedrock-agent", region_name=region, config=config)

# Setup S3 Path to save history audio data
bucket_prefix = 'smart-analysis'
s3_keyframe_path = 'keyframe/'
s3_processed_video_path = 'processed-video/'
bucket_name = ''

#Setup CloudWatch Log
LOG_GROUP_NAME = '/aws/smartAnalysis'
LOG_STREAM_NAME = 'smartAnalysisStream'

# Set the model ID, for example Amazon Titan Text G1 - Express
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


# Initialize the cache variable
bedrock_cache = []
bedrock_frame_count = 0
bedrock_summaries = []

# Key frame buffer
key_frames = deque(maxlen=100)

# Create a directory to store frames
image_dir = "./data/processedVideo"
os.makedirs(image_dir, exist_ok=True)
snapshot_dir = "./data/processedSnapshot"
os.makedirs(snapshot_dir, exist_ok=True)

# Sample image
examples_imgs = [
    # ["data/image/1.png"],
    # ["data/image/2.png"],
    # ["data/image/3.png"],
    # ["data/image/4.png"],
    # ["data/image/5.png"],
    # ["data/image/6.png"],
    # ["data/image/7.png"],
    ["data/image/watermark.png"],
    ["data/image/water.png"],
]
face_collection = [
    ["data/image/jeff.jpg"],
    ["data/image/andy.jpg"],
    ["data/image/andy_portrait.jpg"],
]
# Sample video
examples_videos = [
    ["data/video/person.mp4"],
    ["data/video/bezos_vogels_contentVideo.mp4"],
    ["data/video/livingroomfire.mp4"],
]

##################start feature function##################
def fn_gray(image):
    '''
    Image to grayscale function
    '''
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output

def fn_binary(image):
    '''
    Convert image to binary function
    '''
    gray_image = fn_gray(image)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return binary_image

def fn_salt_and_pepper_noise(image, prob=random.randint(1,100)):
    '''
    Add salt and pepper noise function
    Parameters:
    - image: input image
    - prob: noise probability, i.e. the probability of each pixel being affected by noise
    Return:
    - noisy_image: image with added salt and pepper noise
    '''
    noisy_image = np.copy(image)
    h, w = noisy_image.shape[:2]
    num_noise_pixels = int(prob * h * w)
    
    # Randomly select pixel locations and add noise
    for _ in range(num_noise_pixels):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        noisy_image[y, x] = 0 if np.random.rand() < 0.5 else 255
    
    return noisy_image

def fn_remove_noise(image, kernel_size=random.randint(1,100)):
    '''
    Denoising function
    Parameters:
    - image: input image
    - kernel_size: size of the smoothing filter
    Return:
    - denoised_image: denoised image
    '''
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

def fn_watermark(image: np.ndarray, watermark_path: str = "data/image/water.png") -> np.ndarray:
    '''
    Add watermark function
    Parameters:
    - image: input image 
    - watermark_path: path of the watermark image
    Return:
    - watermarked_image: image with watermark added
    '''
    # Read watermark image
    watermark = cv2.imread(watermark_path)
    if watermark is None:
        raise FileNotFoundError(f"Watermark image not found：{watermark_path}")

    # Check if the image size matches
    if image.shape != watermark.shape:
        raise ValueError("The size of the input image and watermark image do not match.")

    # Add watermark
    watermarked_image = cv2.addWeighted(image, 1, watermark, 0.5, 0)
    return watermarked_image

# Grab the current frame from the camera and display it
def fn_screenshot_webcam(image):
    '''
    Take a screenshot of the camera
    '''
    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, output)
    return image,image

# Grab the fifth frame of the video and display it
def fn_screenshot_frame_5(video):
    '''
    Take a screenshot of the video
    '''
    _video = cv2.VideoCapture(video)
    print("video Type:{}".format(type(video)))
    print(video)
    # Read and discard the first four frames
    for _ in range(4):
        _video.read()
    # Read the fifth frame
    ret, image = _video.read()

    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, image)
    return output,output

# Get and display the current video frame
def fn_screenshot(video):
    '''
    Take a screenshot of the video
    '''
    _video = cv2.VideoCapture(video)
    print("video Type:{}".format(type(video)))
    print(video)
    # Read and discard the first n frames
    for _ in range(random.randint(1,200)):
        _video.read()
    # Read the (n+1)th frame
    ret, image = _video.read()

    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, image)
    return output,output

# Random crop the video
def fn_save_video_random(video_path):
    '''
    Random crop the video
    '''
    output_path = f'video_shotcut_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]split=2[top][bottom];[bottom]crop=iw:ih/2:0:0[bottom1];[top]crop=iw:ih/2:0:ih/2[top1];[top1][bottom1]vstack" -c:a copy "{output_path}"'
    subprocess.call(shell_command,shell=True)
    # Upload the generated video to S3
    s3_client.upload_file(f'./{output_path}', bucket_name, output_path)
    return output_path

# Reverse the video
def fn_video_upend(video_path):
    '''
    Reverse the video
    '''
    output_path = f'video_upend_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -i "{video_path}" -vf "reverse" -af "areverse" "{output_path}"'
    subprocess.call(shell_command, shell=True)
    # Upload the generated video to S3
    s3_client.upload_file(f'./{output_path}', bucket_name, output_path)
    return output_path

# Save a 3 second camera video
def fn_save_video_webcam(input_image):
    '''
    Save a 3 second camera video to a file
    Output: Video path
    '''
    cap = cv2.VideoCapture(0)
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    i = 0
    start_time = time.time()
    while time.time() - start_time < 3:
        image_path = os.path.join(temp_folder, f'image_{i}.jpg')
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_path, frame)
        i += 1
        time.sleep(0.1)  # Pause for 0.1 seconds

    cap.release()

    output_path = f'video_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -f image2 -r 1/3 -i {temp_folder}/image_%d.jpg -vf "fps=10" -y {output_path}'
    subprocess.call(shell_command, shell=True)
    
    shutil.rmtree(temp_folder)
    # Upload the generated video to S3
    s3_client.upload_file(f'./{output_path}', bucket_name, output_path)
    return output_path

def does_bucket_exist(bucket_prefix):
    """Check if there is any bucket with the given prefix."""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.list_buckets()
        for bucket in response['Buckets']:
            if bucket['Name'].startswith(bucket_prefix):
                return bucket['Name']
    except ClientError as e:
        print(f"Error checking bucket existence: {e}")
    return None

def create_unique_bucket(bucket_prefix, region=None):
    """Create an S3 bucket with a unique name."""
    bucket_name = f"{bucket_prefix}-{uuid.uuid4()}"
    print(bucket_name)
    try:
        if region is None or "us-east-1":
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        print(f"Bucket {bucket_name} created successfully.")
        return bucket_name
    except ClientError as e:
        print(f"Error creating bucket: {e}")
        return None
    
def does_folder_exist(bucket_name, folder_name):
    """Check if a folder exists in the specified bucket."""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name + '/')
        for obj in response.get('Contents', []):
            if obj['Key'].startswith(folder_name + '/'):
                return True
    except ClientError as e:
        print(f"Error checking folder existence: {e}")
    return False

def create_folder(bucket_name, folder_name):
    """Create a folder in the specified bucket."""
    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Bucket=bucket_name, Key=(folder_name + '/'))
        print(f"Folder {folder_name} created successfully in bucket {bucket_name}.")
    except ClientError as e:
        print(f"Error creating folder: {e}")

# Create CloudWatch Log Group
def create_log_group():
    try:
        cloudwatchLog.create_log_group(logGroupName=LOG_GROUP_NAME)
    except cloudwatchLog.exceptions.ResourceAlreadyExistsException:
        pass

def create_log_stream():
    try:
        cloudwatchLog.create_log_stream(logGroupName=LOG_GROUP_NAME, logStreamName=LOG_STREAM_NAME)
    except cloudwatchLog.exceptions.ResourceAlreadyExistsException:
        pass

def put_log_events(message):
    response = cloudwatchLog.describe_log_streams(logGroupName=LOG_GROUP_NAME, logStreamNamePrefix=LOG_STREAM_NAME)
    upload_sequence_token = response['logStreams'][0].get('uploadSequenceToken', None)

    log_event = {
        'logGroupName': LOG_GROUP_NAME,
        'logStreamName': LOG_STREAM_NAME,
        'logEvents': [
            {
                'timestamp': int(round(time.time() * 1000)),
                'message': message
            },
        ],
    }

    if upload_sequence_token:
        log_event['sequenceToken'] = upload_sequence_token

    cloudwatchLog.put_log_events(**log_event)

def cleanImage(folder_path):
    entries = os.listdir(folder_path)
    # If the list is empty, it means the folder is empty
    if not entries:
        print("The folder is empty")
    else:
        print("The folder is not empty")
    for entry in entries:
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            os.remove(full_path)
    print("All files have been cleared!")

def fn_open_analysis(video):
    # Clear the previous analysis file
    cleanImage(image_dir)
    video_bytes = cv2.VideoCapture(video)
    output_path = f'video_analysis_{random.randint(100,200)}.mp4'
    # Processing video frames
    frames = process_frames(video,bedrock_cache,bedrock_frame_count)
    # Creating a temporary directory to store frames
    frame_files = []
    # Saving keyframes to S3 and analyzing them
    for i, frame in enumerate(frames):
        # timestamp = frame['timestamp']
        # Invoking AWS services for analysis
        face_detection = rekognition.detect_faces(Image={'Bytes': frame['data']})
        object_detection = rekognition.detect_labels(Image={'Bytes': frame['data']})
        # text_detection = rekognition.detect_text(Image={'Bytes': frame['data']})
        # moderated_content = rekognition.detect_moderation_labels(Image={'Bytes': frame['data']})
        print(face_detection)
        # Drawing bounding boxes on the image
        annotated_frame = draw_bounding_boxes(
            frame['data'], 
            face_detection['FaceDetails'], 
            object_detection['Labels']
        )
        # Save the processed image with red bounding boxes to the directory
        local_frame_path = os.path.join(image_dir, f"frame{i:04d}.jpg")
        with open(local_frame_path, 'wb') as f:
            f.write(annotated_frame)
        frame_files.append(local_frame_path)

        # Upload the processed image with red bounding boxes to S3
        image_id = save_frame_to_s3(frame['data'],s3_keyframe_path)
            
         # Convert the frame data to bytes
        frame_bytes = bytes(frame['data'])
        
        # Use the base64 module for encoding
        frame_base64 = base64.b64encode(frame_bytes)
            
        # Decode the encoded data into a string
        frame_base64_str = frame_base64.decode('utf-8')
    
    # Use FFmpeg to combine videos
    video_name = f"output_{int(time.time())}.mp4"
    
    cmd = ['ffmpeg','-framerate','1', '-i', './data/processedVideo/frame%4d.jpg', video_name]
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))    

    # Uploading the generated video to S3
    videoname_on_s3 = s3_processed_video_path+video_name
    s3_client.upload_file(f'./{video_name}', bucket_name, videoname_on_s3)
    
    # Summarize and provide warnings for video content
    native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You need to summarize based on the following text, which is a collection of video frame captures. You should summarize the content happening \
                                    in these frames, such as what happened, whether there was a fire, or if there were any suspicious people trying to steal things. If any of these \
                                    dangerous situations occur whatever it is real or not, you need to call an agent and inform the agent of the dangerous information. The text is as follows:".join(bedrock_summaries)
                        }],
                }
            ],
        }
        # Convert the native request to JSON.
    requestJson = json.dumps(native_request)
    print(requestJson)
    response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                body=requestJson
        )

    # Decode the response body.
    model_response = json.loads(response["body"].read())
    # Extract and print the response text.
    bedrock_summary = model_response["content"][0]["text"]
    put_log_events(f'Finish Bedrock Summary{bedrock_summary}')
    print(bedrock_summary)
    
    agentId = bedrock_agent.list_agents()['agentSummaries'][0]['agentId']
    agentAliasId = bedrock_agent.list_agent_aliases(agentId=agentId)['agentAliasSummaries'][0]['agentAliasId']
    # Invoke Bedrock agent
    response = bedrock_agent_runtime.invoke_agent(
    agentId=agentId,      # Identifier for Agent
    agentAliasId=agentAliasId, # Identifier for Agent Alias
    sessionId='session123',    # Identifier used for the current session
    inputText=bedrock_summary)

    output = ""

    stream = response.get('completion')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                output += chunk.get('bytes').decode()
    put_log_events(f'Finish Bedrock {bedrock_summary}')
    return video_name, bedrock_summary

#Analysis by Bedrock LLM
def analysis_by_llm(image_cache):
                    # Build a chat message containing text and images
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 512,
                    "temperature": 0.5,
                    "messages": [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]
                    }
                    
                    # Add all images
                for frame_data in image_cache:
                    frame_base64 = base64.b64encode(frame_data).decode('utf-8')
                    native_request["messages"][0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_base64
                        }
                    })
            
                # Add text prompt
                native_request["messages"][0]["content"].append(
                    {
                        "type": "text",
                        "text": "You need to generate an overall introduction based on these photos, describing what happened in the images, whether there were any abnormal situations such as a \
                                fire or suspicious individuals, approximately 50-100 words."
                    }
                )
                
                # Convert the native request to JSON.
                requestJson = json.dumps(native_request)

                response = bedrock.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    body=requestJson
                )

                # Decode the response body.
                model_response = json.loads(response["body"].read())
                # Extract and print the response text.
                bedrock_summary = model_response["content"][0]["text"]
                return bedrock_summary

def process_frames(video_path,bedrock_cache, bedrock_frame_count):
    frames = []
    # Read video from temp file
    cap = cv2.VideoCapture(video_path)
    # Get video file resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video resolution: {width}x{height}")
    
    # Get video file frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video frame rate: {fps}")
    
    
   # Calculate the interval between frames (in milliseconds)
    frame_interval = int(500)  # 0.5秒

    frame_count = 0
    current_frame_time = 0

    while True:
        # Jump directly to the next position where a frame needs to be extracted
        current_frame_time += frame_interval
        cap.set(cv2.CAP_PROP_POS_MSEC, current_frame_time)

        ret, frame = cap.read()

        if frame is None:
            print("Unable to read image data or have already finished processing")
            break

        # Remove duplicate frames
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_bytes = img.tobytes()
        frame_hash = hash(frame_bytes)

        if frame_hash not in key_frames:
            key_frames.append(frame_hash)
            timestamp = datetime.now().timestamp()
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append({'data': buffer.tobytes(), 'timestamp': timestamp})
            
            # Cache frame data to bedrock_cache
            bedrock_cache.append(buffer.tobytes())
            bedrock_frame_count += 1
            
            # Call bedrock once every 5 frames to get the summary of these 5 frames
            if bedrock_frame_count == 5:
                # Extract and print the response text.
                bedrock_summary = analysis_by_llm(bedrock_cache)
                print(bedrock_summary)

                # Add the current batch's bedrock_summary to the list
                bedrock_summaries.append(bedrock_summary)

                # Clear the cache variables
                bedrock_cache = []
                bedrock_frame_count = 0

        frame_count += 1

    cap.release()
    print(f"Totally Processed {frame_count} frames!")
    return frames
    
    
def draw_bounding_boxes(image_bytes, faces, labels):
    # Open the image
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)

    # Get the image dimensions
    width, height = image.size

    # Draw face bounding boxes
    for face in faces:
        box = face['BoundingBox']
        left = width * box['Left']
        top = height * box['Top']
        right = left + (width * box['Width'])
        bottom = top + (height * box['Height'])
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

    # Draw label bounding boxes
    for label in labels:
        for instance in label.get('Instances', []):
            box = instance['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            right = left + (width * box['Width'])
            bottom = top + (height * box['Height'])
            draw.rectangle([left, top, right, bottom], outline="blue", width=2)
            draw.text((left, top - 10), label['Name'], fill="blue")

    # Convert the image back to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    print("Image drawing completed")
    return buffered.getvalue()
    
    
# Save the frame to S3
def save_frame_to_s3(frame,s3_path):
    image_id = str(uuid.uuid4())
    s3_key = s3_path + f"frame_{image_id}.jpg"
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=frame)
    return image_id


# Camera screenshot analysis
def fn_screenshot_analysis(image: np.ndarray) -> np.ndarray:
    snapshot_cache = []
    # Encode the NumPy array in JPEG format
    _, jpeg_data = cv2.imencode('.jpg', image)
 
    print(type(jpeg_data))

    jpeg_data_bytes = jpeg_data.tobytes()
     # Call AWS services for analysis
    face_detection = rekognition.detect_faces(Image={'Bytes': jpeg_data_bytes})
    object_detection = rekognition.detect_labels(Image={'Bytes': jpeg_data_bytes})
    text_detection = rekognition.detect_text(Image={'Bytes': jpeg_data_bytes})
    print(face_detection)
    
    # Draw bounding boxes on the image
    annotated_frame = draw_bounding_boxes(
        jpeg_data_bytes, 
        face_detection['FaceDetails'], 
        object_detection['Labels']
        )
    
    analyzied_snapshot_name = f"snapshot_analysis_{int(time.time())}.jpg"
    
    # Save the processed image with red boxes to the directory
    local_frame_path = os.path.join(snapshot_dir, analyzied_snapshot_name)
    with open(local_frame_path, 'wb') as f:
        f.write(annotated_frame)
    snapshot_cache.append(jpeg_data_bytes)
    bedrock_results = analysis_by_llm(snapshot_cache)

    # Clear the cache
    snapshot_cache=[]
    
    # Return the screenshot with red boxes, with the bedrock analysis results
    return local_frame_path,bedrock_results
    

def create_collection(collection_id):

    # Create a collection
    print('Creating collection:' + collection_id)
    response = rekognition.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')


def add_faces_to_collection(photo_path, collection_id):
    print(photo_path)
    
    imageTarget = open(photo_path, 'rb')

    ExternalImageId = ''.join(random.choices(string.digits, k= 6 ))

    response = rekognition.index_faces(CollectionId=collection_id,
                                  Image={'Bytes': imageTarget.read()},
                                  ExternalImageId=ExternalImageId,
                                  MaxFaces=1,
                                  QualityFilter="AUTO",
                                  DetectionAttributes=['ALL'])

    print('Results for ' + photo_path)
    for faceRecord in response['FaceRecords']:
        print('  Face ID: ' + faceRecord['Face']['FaceId'])

    return response


def create_user(collection_id, user_id):
    
        # Check if the user-id exists, if not, create a new user
    try:
        user_response = rekognition.list_users(CollectionId=collection_id)
        existing_users = user_response['Users']
        next_token = user_response.get('NextToken')
    
        # If there are more users to get, continue the loop
        while next_token:
            user_response = rekognition.list_users(CollectionId=collection_id, NextToken=next_token)
            existing_users.extend(user_response['Users'])
            next_token = user_response.get('NextToken')
    
        # Check if the user to be created already exists
        if any(user['UserId'] == user_id for user in existing_users):
            print(f"User {user_id} already exists in collection {collection_id}")
        else:
            # If the user doesn't exist, create a new user
            rekognition.create_user(
            CollectionId=collection_id,
            UserId=user_id
        )
            print(f"Creating user {user_id} in collection {collection_id}")
    except rekognition.exceptions.ResourceNotFoundException:
        print(f"Collection {collection_id} does not exist")
    except Exception as e:
        print(f"Error: {e}")

def associate_faces(collection_id, user_id, face_ids):

    logger.info(f'Associating faces to user: {user_id}, {face_ids}')
    try:
        response = rekognition.associate_faces(
            CollectionId=collection_id,
            UserId=user_id,
            FaceIds=face_ids
        )
        print(f'- associated {len(response["AssociatedFaces"])} faces')
    except ClientError:
        logger.exception("Failed to associate faces to the given user")
    else:
        print(response)
        return response
        
    
# Camera screenshot analysis
def fn_face_comparison(image: np.ndarray) -> np.ndarray:
    collection_id = 'videoAnalysis'
    
    # Apart from the first time, remember to change this \
    # ID to a new value each time, otherwise repeated submission of the same user-id will report a parameter type error.
    user_id = '123456'

    snapshot_cache = []
    # Encode NumPy array into JPEG format
    _, jpeg_data = cv2.imencode('.jpg', image)

    print(type(jpeg_data))
    jpeg_data_bytes = jpeg_data.tobytes()
    
    
    
    # Check if the collection already exists
    try:
        response = rekognition.describe_collection(CollectionId=collection_id)
        print(f'Collection {collection_id} already exists.')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Create a new collection if it doesn't exist
            create_collection(collection_id)
        else:
            raise e

    # Add local images to the collection
    photos = ['data/image/andy.jpg','data/image/jeff1.jpg','data/image/jeff2.jpg','data/image/jeff3.jpg',"data/image/andy_portrait.jpg"]
    face_ids = []
    for photo in photos:
        response = add_faces_to_collection(photo, collection_id)
        if  response:
            if 'FaceRecords' in response:
                for face_record in response['FaceRecords']:
                    face_id = face_record['Face']['FaceId']
                    face_ids.append(face_id)
            else:
                print(f"No faces detected in {photo}")
    print(face_ids)
    
    
    # Create a user
    create_user(collection_id, user_id)
    
    # Associate faces with the user
    # associate_faces(collection_id, user_id, face_ids)
        
    # Compare the screenshot photo sent against the database
    try:
        search_response = rekognition.search_users_by_image(
            CollectionId=collection_id,
            Image={'Bytes': jpeg_data_bytes}
        )
        
        print(type(search_response))
        result_dict = search_response
        # Check if there is a user match
        if 'UserMatches' in result_dict and len(result_dict['UserMatches']) > 0:
            # Get the first user match result
            user_match = result_dict['UserMatches'][0]
    
            # Check if the user status is active
            if user_match['User']['UserStatus'] == 'ACTIVE':
                return "User match succeeded, unlock!"
            else:
                return "User status is inactive, please try again!"
        else:
            return "User match failed, please try again!"
            

    except ClientError:
        logger.exception(f'Failed to perform SearchUsersByImage with given image: {jpeg_data}')
        raise
    else:
        print(response)
        return response
    
##################End function####################
# Setup Python Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create CloudWatch log group and stream
create_log_group()
create_log_stream()

try:
    logger.info("This is an info message")
    put_log_events("This is an info message")
except (NoCredentialsError, PartialCredentialsError) as e:
    logger.error("AWS credentials not found or incomplete: %s", e)

# Check if a bucket with the specified prefix exists
bucket_name = does_bucket_exist(bucket_prefix)
print(bucket_name)
if bucket_name:
    print(f"A bucket with prefix '{bucket_prefix}' already exists: {bucket_name}")
else:
    # Create a new unique bucket
    bucket_name = create_unique_bucket(bucket_prefix, region)
    print(bucket_name)
if bucket_name:
    # Check if the folder exists in the bucket
    if does_folder_exist(bucket_name, s3_processed_video_path):
        print(f"Folder '{s3_processed_video_path}' already exists in bucket '{bucket_name}'.")
    else:
        # Create the folder in the bucket
        create_folder(bucket_name, s3_processed_video_path)
    # Check if the folder exists in the bucket
    if does_folder_exist(bucket_name, s3_keyframe_path):
        print(f"Folder '{s3_keyframe_path}' already exists in bucket '{bucket_name}'.")
    else:
        # Create the folder in the bucket
        create_folder(bucket_name, s3_keyframe_path)

##################start Interface building##################

# 构建界面Blocks上下文
with gradio.Blocks() as demo:
    gradio.Markdown("# Smart Multi-Mode Analysis & Alarm")
    gradio.Markdown("## Image Processing")
    
    # Original image preview and processed image preview
    # Arranged vertically
    with gradio.Column():
        # Arranged horizontally
        with gradio.Row():
            before_img = gradio.Image(label="Original Image")
            after_img = gradio.Image(label="Processed Image")
        # Arranged horizontally
        with gradio.Row():
            gradio.Examples(examples=examples_imgs, inputs=[before_img],label="Example Images")
        with gradio.Row(elem_id="Image Processing Function"):   
            fn_gray_btn = gradio.Button("Grayscale Image")
            fn_binary_btn = gradio.Button("Binary Image")
            fn_salt_and_pepper_noise_btn = gradio.Button("Add Salt and Pepper Noise")
            fn_watermark_btn = gradio.Button("Add a Frequency Domain Watermark")    

        fn_gray_btn.click(fn=fn_gray, inputs=[before_img], outputs=after_img)
        fn_binary_btn.click(fn=fn_binary, inputs=[before_img], outputs=after_img)
        fn_salt_and_pepper_noise_btn.click(fn=fn_salt_and_pepper_noise, inputs=[before_img], outputs=after_img)
        fn_watermark_btn.click(fn=fn_watermark, inputs=[before_img], outputs=after_img)

    gradio.Markdown("## Video Processing")
    with gradio.Column():
        with gradio.Row():
            before_video = gradio.Video(label="Original Video")
            before_video_webcam = gradio.Image(sources="webcam", streaming=True,label="Camera Preview",visible=True)
            after_video = gradio.Video(label="Processed Video")
            shotcut_video = gradio.Image(label="Video Screenshot")
        gradio.Markdown("Video Operation Area")
        with gradio.Row():
            fn_screenshot_btn = gradio.Button("Video Screenshot")
            fn_screenshot_webcam_btn = gradio.Button("Camera Screenshot")
            fn_save_video_webcam_btn = gradio.Button("Save Camera Video")
            fn_screenshot_frame_5_btn = gradio.Button("Take Screenshot of the 5th Frame")
            fn_video_upend_btn = gradio.Button("Reverse Playback")
            fn_open_analysis_btn = gradio.Button("Video Intelligent Analysis")
            fn_shutcut_analysis_btn = gradio.Button("Screenshot Analysis")
            fn_face_comparison_btn = gradio.Button("Face Verification")

        with gradio.Row():
            gradio.Examples(examples=examples_videos, inputs=[before_video], label="Sample Video")
            shotcut_analysis_video = gradio.Image(label="Video Screenshot Analysis - Illustrated")
            shotcut_analysis_text = gradio.Textbox(label="Video Screenshot Analysis - Textual", lines=4, placeholder="Click the button to start analysis...",)
            face_comparison_text = gradio.Textbox(label="Face Verification Result", lines=4, placeholder="Click the button to start analysis...",)
            
        with gradio.Row():
            video_smart_analysis_result_text = gradio.Textbox(label="Video Intelligent Analysis Result", lines=4, placeholder="Click the button to start analysis....",)

        face_how = gradio.Image(label="Face Display")
        with gradio.Row():
            gradio.Examples(examples=face_collection, inputs=[face_how], label="Face Collection")

        fn_open_analysis_btn.click(fn=fn_open_analysis,inputs=[before_video],outputs=[after_video,video_smart_analysis_result_text])
        fn_shutcut_analysis_btn.click(fn=fn_screenshot_analysis,inputs=[shotcut_video],outputs=[shotcut_analysis_video,shotcut_analysis_text])
        fn_face_comparison_btn.click(fn=fn_face_comparison,inputs=[shotcut_video],outputs=[face_comparison_text])

        fn_screenshot_btn.click(fn=fn_screenshot,inputs=[before_video],outputs=[shotcut_video,before_img])
        fn_screenshot_webcam_btn.click(fn=fn_screenshot_webcam,inputs=[before_video_webcam],outputs=[shotcut_video,before_img])
        fn_screenshot_frame_5_btn.click(fn=fn_screenshot_frame_5,inputs=[before_video],outputs=[shotcut_video,before_img])
        fn_video_upend_btn.click(fn=fn_video_upend,inputs=[before_video],outputs=[after_video])
        fn_save_video_webcam_btn.click(fn=fn_save_video_webcam,inputs=[before_video_webcam],outputs=[after_video])

# Launch demo interface
demo.launch(share=True)

##################End Interface construction##################