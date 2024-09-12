import cv2,boto3,gradio,random,os,json,io,string
import subprocess,time,shutil,base64,uuid,logging
import numpy as np
from collections import deque
from PIL import Image,ImageDraw
from datetime import datetime 
# from decimal import Decimal
from botocore.exceptions import ClientError,NoCredentialsError, PartialCredentialsError

#指定可用区
region = 'us-east-1'

# AWS 客户端初始化
session = boto3.Session()
s3_client = session.client('s3',region_name=region)
dynamodb = session.resource('dynamodb',region_name=region)
table = dynamodb.Table('videoInfo')
rekognition = session.client('rekognition',region_name=region)
bedrock = boto3.client('bedrock-runtime',region_name=region)
cloudwatchLog = boto3.client('logs',region_name=region)

# Setup S3 Path to save history audio data
bucket_prefix = 'smart-analysis'
s3_keyframe_path = 'keyframe/'
s3_processed_video_path = 'processed-video/'
bucket_name = ''

#Setup CloudWatch Log
LOG_GROUP_NAME = '/aws/smartAnalysis'
LOG_STREAM_NAME = 'smartAnalysisStream'

# 设置模型 ID,例如 Amazon Titan Text G1 - Express
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


# 初始化缓存变量
bedrock_cache = []
bedrock_frame_count = 0
bedrock_summaries = []

# 关键帧缓冲区
key_frames = deque(maxlen=100)

# 创建目录来存储帧
image_dir = "./data/processedVideo"
os.makedirs(image_dir, exist_ok=True)
snapshot_dir = "./data/processedSnapshot"
os.makedirs(snapshot_dir, exist_ok=True)

# 示例图片
examples_imgs = [
    ["data/image/1.png"],
    ["data/image/2.png"],
    ["data/image/3.png"],
    ["data/image/4.png"],
    ["data/image/5.png"],
    ["data/image/6.png"],
    ["data/image/7.png"],
    ["data/image/watermark.png"],
    ["data/image/water.png"],
]
face_collection = [
    ["data/image/jeff.jpg"],
    ["data/image/andy.jpg"],
    ["data/image/andy_portrait.jpg"],
]
# 示例视频
examples_videos = [
    ["data/video/person.mp4"],
    ["data/video/bezos_vogels_contentVideo.mp4"],
    ["data/video/livingroomfire.mp4"],
]

##################start 功能函数##################
def fn_gray(image):
    '''
    图片转灰度函数
    '''
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output

def fn_binary(image):
    '''
    将图像转换为二值图函数
    '''
    gray_image = fn_gray(image)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return binary_image

def fn_salt_and_pepper_noise(image, prob=random.randint(1,100)):
    '''
    添加椒盐噪声函数
    参数：
    - image: 输入的图像
    - prob: 噪声概率,即每个像素被噪声影响的概率
    返回：
    - noisy_image: 添加了椒盐噪声的图像
    '''
    noisy_image = np.copy(image)
    h, w = noisy_image.shape[:2]
    num_noise_pixels = int(prob * h * w)
    
    # 随机选择像素位置并添加噪声
    for _ in range(num_noise_pixels):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        noisy_image[y, x] = 0 if np.random.rand() < 0.5 else 255
    
    return noisy_image

def fn_remove_noise(image, kernel_size=random.randint(1,100)):
    '''
    去除噪声函数
    参数：
    - image: 输入的图像
    - kernel_size: 平滑滤波器的大小
    返回：
    - denoised_image: 去除噪声后的图像
    '''
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

def fn_watermark(image: np.ndarray, watermark_path: str = "data/image/water.png") -> np.ndarray:
    '''
    添加水印函数
    参数：
    - image: 输入的图像
    - watermark_path: 水印图像的路径
    返回：
    - watermarked_image: 添加水印后的图像
    '''
    # 读取水印图像
    watermark = cv2.imread(watermark_path)
    if watermark is None:
        raise FileNotFoundError(f"水印图像未找到：{watermark_path}")

    # 检查图像大小是否匹配
    if image.shape != watermark.shape:
        raise ValueError("输入图像和水印图像的大小不匹配")

    # 添加水印
    watermarked_image = cv2.addWeighted(image, 1, watermark, 0.5, 0)
    return watermarked_image

# 获取摄像头当前帧并显示
def fn_screenshot_webcam(image):
    '''
    截图摄像头
    '''
    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, output)
    return image,image

# 获取视频第五帧并显示
def fn_screenshot_frame_5(video):
    '''
    截图视频
    '''
    _video = cv2.VideoCapture(video)
    print("video类型:{}".format(type(video)))
    print(video)
    # 读取并丢弃前四帧
    for _ in range(4):
        _video.read()
    # 读取第五帧
    ret, image = _video.read()

    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, image)
    return output,output

# 获取视频当前帧并显示
def fn_screenshot(video):
    '''
    截图视频
    '''
    _video = cv2.VideoCapture(video)
    print("video类型:{}".format(type(video)))
    print(video)
    # 读取并丢弃前n帧
    for _ in range(random.randint(1,200)):
        _video.read()
    # 读取第n+1帧
    ret, image = _video.read()

    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = "screenshot_{}.jpg".format(random.randint(100,200))
    cv2.imwrite(file_name, image)
    return output,output

# 随机剪辑视频
def fn_save_video_random(video_path):
    '''
    随机剪辑视频
    '''
    output_path = f'video_shotcut_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]split=2[top][bottom];[bottom]crop=iw:ih/2:0:0[bottom1];[top]crop=iw:ih/2:0:ih/2[top1];[top1][bottom1]vstack" -c:a copy "{output_path}"'
    subprocess.call(shell_command,shell=True)
    # 上传生成的视频到 S3
    s3_client.upload_file(f'./{output_path}', bucket_name, output_path)
    return output_path

# 视频倒放
def fn_video_upend(video_path):
    '''
    视频倒放
    '''
    output_path = f'video_upend_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -i "{video_path}" -vf "reverse" -af "areverse" "{output_path}"'
    subprocess.call(shell_command, shell=True)
    # 上传生成的视频到 S3
    s3_client.upload_file(f'./{output_path}', bucket_name, output_path)
    return output_path

# 保存3s摄像头视频
def fn_save_video_webcam(input_image):
    '''
    保存3s摄像头视频到文件
    输出:视频路径
    '''
    cap = cv2.VideoCapture(0)
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    i = 0
    start_time = time.time()
    while time.time() - start_time < 3:
        image_path = os.path.join(temp_folder, f'image_{i}.jpg')
        #pil_image = Image.fromarray(np.uint8(input_image))
        #pil_image.save(image_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_path, frame)
        i += 1
        time.sleep(0.1)  # 暂停0.1秒钟

    cap.release()

    output_path = f'video_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -f image2 -r 1/3 -i {temp_folder}/image_%d.jpg -vf "fps=10" -y {output_path}'
    subprocess.call(shell_command, shell=True)
    
    shutil.rmtree(temp_folder)
    # 上传生成的视频到 S3
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
    # 如果列表为空，说明文件夹为空
    if not entries:
        print("文件夹为空")
    else:
        print("文件夹不为空")
    for entry in entries:
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            os.remove(full_path)
    print("已清空所有文件！")

def fn_open_analysis(video):
    #清除上次分析文件
    cleanImage(image_dir)
    # print(video)
    video_bytes = cv2.VideoCapture(video)
    #print("video类型:{}".format(type(video)))

    output_path = f'video_analysis_{random.randint(100,200)}.mp4'
    
    # 处理视频帧
    frames = process_frames(video,bedrock_cache,bedrock_frame_count)
    
    
    
    # 创建临时目录来存储帧
    # with tempfile.TemporaryDirectory() as temp_dir:
    frame_files = []
        
    # 保存关键帧到 S3 并分析
    for i, frame in enumerate(frames):
            
        timestamp = frame['timestamp']
        # 调用 AWS 服务进行分析
        face_detection = rekognition.detect_faces(Image={'Bytes': frame['data']})
        object_detection = rekognition.detect_labels(Image={'Bytes': frame['data']})
        text_detection = rekognition.detect_text(Image={'Bytes': frame['data']})
        moderated_content = rekognition.detect_moderation_labels(Image={'Bytes': frame['data']})
        print(face_detection)
        # 在图像上绘制边界框
        annotated_frame = draw_bounding_boxes(
            frame['data'], 
            face_detection['FaceDetails'], 
            object_detection['Labels']
        )
        # 将处理好的带红标图像保存到目录
        local_frame_path = os.path.join(image_dir, f"frame{i:04d}.jpg")
        with open(local_frame_path, 'wb') as f:
            f.write(annotated_frame)
        frame_files.append(local_frame_path)

        # 将处理好的带红标图像上传到s3
        image_id = save_frame_to_s3(frame['data'],s3_keyframe_path)
            
         # 将 frame 数据转换为 bytes
        frame_bytes = bytes(frame['data'])
        
        # 使用 base64 模块进行编码
        frame_base64 = base64.b64encode(frame_bytes)
            
        # 将编码后的数据解码为字符串
        frame_base64_str = frame_base64.decode('utf-8')
            
    #         guardrail_filter = "test"
    # #        guardrail_filter = bedrock.detect_pii_entities(Text=bedrock_summary['Summary'])
    
    #         Item={
    #                 'imageID': image_id,
    #                 's3URL': f"https://cf.haozhiyu.fun/test/frame_{image_id}",
    #                 'timestamp': timestamp,
    #                 'faceDetection': face_detection,
    #                 'objectsDetection': object_detection,
    #                 'textDetection': text_detection,
    #                 'moderatedContent': moderated_content,
    #                 'guardrailFilter': guardrail_filter
    #         }
            
            # timestamp = str(timestamp)
            # image_id = json.dumps(image_id)
            # faceDetection = json.dumps(face_detection)
            # s3URL = str(f"https://cf.haozhiyu.fun/test/frame_{image_id}")
            # objectsDetection = json.dumps(object_detection)
            # textDetection = json.dumps(text_detection)
            # moderatedContent = json.dumps(moderated_content)
            #bedrockSummary = json.dumps(bedrock_summary)
            
            # item = json.loads(json.dumps(Item), parse_float=Decimal)
    
            # table.put_item(
            #     Item=item
            # )
    
    # 使用 FFmpeg 合成视频
    video_name = f"output_{int(time.time())}.mp4"
    
    cmd = ['ffmpeg','-framerate','1', '-i', './data/processedVideo/frame%4d.jpg', video_name]
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))    

    # 上传生成的视频到 S3
    #video_s3_key = f"processed_videos/output_{int(time.time())}.mp4"
    videoname_on_s3 = s3_processed_video_path+video_name
    s3_client.upload_file(f'./{video_name}', bucket_name, videoname_on_s3)
    
    #视频内容总结以及警告
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
                            "text": "你需要根据如下文字进行总结，这些文字是视频的抽帧的集合，你要把这些帧发生的内容做统一的总结，比如发生了什么，是不是着火了，或者有可疑人来偷东西。如果出现这些任何危险的事情，需要调用agent，将危险信息告诉agent，文字如下:".join(bedrock_summaries)
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
    
    Item_bedrock={
                'imageID': "111",
                'bedrock_summary': bedrock_summary
        }
        
    # table.put_item(
    #         Item=Item_bedrock
    #     )
    return video_name,bedrock_summary

#Analysis by Bedrock LLM
def analysis_by_llm(image_cache):
                    # 构建对话消息,包含文本和图片
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
                    
                    # 添加所有图片
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
            
                # 添加文本提示
                native_request["messages"][0]["content"].append({
                    "type": "text",
                    "text": "你需要根据这些照片来生成一个总体介绍，描述图片中发生了什么，是否有异常情况如着火或可疑人员，大约50-100字"
                })
                
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
    """# 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name"""

    # 从临时文件中读取视频
    cap = cv2.VideoCapture(video_path)

    
    # 获取视频文件的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频分辨率: {width}x{height}")
    
    # 获取视频文件的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps}")
    
    
   # 计算抽帧间隔(以毫秒为单位)
    frame_interval = int(500)  # 0.5秒

    frame_count = 0
    current_frame_time = 0

    while True:
        # 直接跳转到下一个需要抽取的帧位置
        current_frame_time += frame_interval
        cap.set(cv2.CAP_PROP_POS_MSEC, current_frame_time)

        ret, frame = cap.read()

        if frame is None:
            print("无法读取图像数据或已经处理完毕")
            break

        # 去重帧
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_bytes = img.tobytes()
        frame_hash = hash(frame_bytes)

        if frame_hash not in key_frames:
            key_frames.append(frame_hash)
            timestamp = datetime.now().timestamp()
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append({'data': buffer.tobytes(), 'timestamp': timestamp})
            
            # 缓存帧数据到 bedrock_cache
            bedrock_cache.append(buffer.tobytes())
            bedrock_frame_count += 1
            
            # 每 5 帧调用一次 bedrock,去获取这5帧的总结
            if bedrock_frame_count == 5:
                # Extract and print the response text.
                bedrock_summary = analysis_by_llm(bedrock_cache)
                print(bedrock_summary)

                # 将当前批次的 bedrock_summary 添加到列表中
                bedrock_summaries.append(bedrock_summary)

                # 清空缓存变量
                bedrock_cache = []
                bedrock_frame_count = 0

        frame_count += 1

    cap.release()
#    os.remove(temp_file)
    print(f"总共处理了 {frame_count} 帧")
    return frames
    
    
def draw_bounding_boxes(image_bytes, faces, labels):
    # 打开图像
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)

    # 获取图像尺寸
    width, height = image.size

    # 绘制人脸边界框
    for face in faces:
        box = face['BoundingBox']
        left = width * box['Left']
        top = height * box['Top']
        right = left + (width * box['Width'])
        bottom = top + (height * box['Height'])
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

    # 绘制标签边界框
    for label in labels:
        for instance in label.get('Instances', []):
            box = instance['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            right = left + (width * box['Width'])
            bottom = top + (height * box['Height'])
            draw.rectangle([left, top, right, bottom], outline="blue", width=2)
            draw.text((left, top - 10), label['Name'], fill="blue")

    # 将图像转换回字节
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    print("Image drawing completed")
    return buffered.getvalue()
    
    
# 保存帧到 S3
def save_frame_to_s3(frame,s3_path):
    image_id = str(uuid.uuid4())
    s3_key = s3_path + f"frame_{image_id}.jpg"
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=frame)
    return image_id


#摄像头截图分析
def fn_screenshot_analysis(image: np.ndarray) -> np.ndarray:
    snapshot_cache = []
    # 将NumPy数组编码为JPEG格式
    _, jpeg_data = cv2.imencode('.jpg', image)
 
    print(type(jpeg_data))

    jpeg_data_bytes = jpeg_data.tobytes()
     # 调用 AWS 服务进行分析
    face_detection = rekognition.detect_faces(Image={'Bytes': jpeg_data_bytes})
    object_detection = rekognition.detect_labels(Image={'Bytes': jpeg_data_bytes})
    text_detection = rekognition.detect_text(Image={'Bytes': jpeg_data_bytes})
    #moderated_content = rekognition.detect_moderation_labels(Image={'Bytes': shotcut_video)
    print(face_detection)
    
    # 在图像上绘制边界框
    annotated_frame = draw_bounding_boxes(
        jpeg_data_bytes, 
        face_detection['FaceDetails'], 
        object_detection['Labels']
        )
    
    analyzied_snapshot_name = f"snapshot_analysis_{int(time.time())}.jpg"
    
    # 将处理好的带红标图像保存到目录
    local_frame_path = os.path.join(snapshot_dir, analyzied_snapshot_name)
    with open(local_frame_path, 'wb') as f:
        f.write(annotated_frame)
    snapshot_cache.append(jpeg_data_bytes)
    bedrock_results = analysis_by_llm(snapshot_cache)

    # 清理缓存
    snapshot_cache=[]
    
    #返回带红框截图,bedrock分析结果
    return local_frame_path,bedrock_results
    

def create_collection(collection_id):

    # Create a collection
    print('Creating collection:' + collection_id)
    response = rekognition.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')


def add_faces_to_collection(photo_path, collection_id):

    with open(photo_path, 'rb') as image_file:
        image_bytes = image_file.read()

    ExternalImageId = ''.join(random.choices(string.digits, k= 6 ))

    response = rekognition.index_faces(CollectionId=collection_id,
                                  Image={'Bytes': image_bytes},
                                  ExternalImageId=ExternalImageId,
                                  MaxFaces=1,
                                  QualityFilter="AUTO",
                                  DetectionAttributes=['ALL'])

    print('Results for ' + photo_path)
    print('Faces indexed:')
    for faceRecord in response['FaceRecords']:
        print('  Face ID: ' + faceRecord['Face']['FaceId'])
        print('  Location: {}'.format(faceRecord['Face']['BoundingBox']))

    print('Faces not indexed:')
    for unindexedFace in response['UnindexedFaces']:
        print(' Location: {}'.format(unindexedFace['FaceDetail']['BoundingBox']))
        print(' Reasons:')
        for reason in unindexedFace['Reasons']:
            print('   ' + reason)
    return len(response['FaceRecords'])


def create_user(collection_id, user_id):
    
    try:
        rekognition.create_user(
            CollectionId=collection_id,
            UserId=user_id
        )
    except ClientError:
        logger.exception(f'Failed to create user with given user id: {user_id}')
        raise


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
        raise
    else:
        print(response)
        return response
        
    
#摄像头截图分析
def fn_face_comparison(image: np.ndarray) -> np.ndarray:
    collection_id = 'videoAnalysis'
    user_id = '000001'

    snapshot_cache = []
    # 将NumPy数组编码为JPEG格式
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
    photos = ['data/image/andy.jpg', 'data/image/andyjassy.jpg','data/image/andyjassy2.jpg']
    face_ids = []
    for photo in photos:
        num_faces = add_faces_to_collection(photo, collection_id)
        print(num_faces)
        if num_faces > 0:
            if 'FaceRecords' in response:
                for face_record in response['FaceRecords']:
                    face_id = face_record['Face']['FaceId']
                    face_ids.append(face_id)
                    print(face_ids)
            else:
                print(f"No faces detected in {photo}")

    # Create a new user
    create_user(collection_id, user_id)

    # Associate faces with the user
    associate_faces(collection_id, user_id, face_ids)
        
    #查询截图传来的照片，做比对
    try:
        response = rekognition.search_users_by_image(
            CollectionId=collection_id,
            Image={'Bytes': jpeg_data_bytes}
        )
        
        result_dict = json.loads(response)

        # 检查是否有用户匹配
        if 'UserMatches' in result_dict and len(result_dict['UserMatches']) > 0:
            # 获取第一个用户匹配结果
            user_match = result_dict['UserMatches'][0]
    
            # 检查用户状态是否为活跃状态
            if user_match['User']['UserStatus'] == 'ACTIVE':
                return "用户比对成功,开锁"
            else:
                return "用户状态不活跃,请重试"
        else:
            return "用户比对失败,请重试"
            

    except ClientError:
        logger.exception(f'Failed to perform SearchUsersByImage with given image: {jpeg_data}')
        raise
    else:
        print(response)
        return response
    
##################end 功能函数####################
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
    # Check if the folder exists in the bucket
    # if does_folder_exist(bucket_name, s3_Audio_path):
    #     print(f"Folder '{s3_Audio_path}' already exists in bucket '{bucket_name}'.")
    # else:
    #     # Create the folder in the bucket
    #     create_folder(bucket_name, s3_Audio_path)

##################start 界面构建##################

# 构建界面Blocks上下文
with gradio.Blocks() as demo:
    gradio.Markdown("# SmartMultiModeAnalysis")
    gradio.Markdown("## 图像处理")
    
    # 原图预览和处理后图像预览
    # 纵向排列
    with gradio.Column():
        # 横向排列
        with gradio.Row():  
            # before_img = gradio.Image(label="原图",height=200, width=200)
            # after_img = gradio.Image(label="处理后图片",height=200, width=200)
            before_img = gradio.Image(label="原图")
            after_img = gradio.Image(label="处理后图片")
            #interface_img = gradio.Interface(fn=fun_gray_img, inputs="image", outputs="image")
        # 横向排列
        with gradio.Row():
            gradio.Examples(examples=examples_imgs, inputs=[before_img],label="示例图片")
        # 横向排列
        with gradio.Row(elem_id="图片处理函数"):   
            # 按钮:转为灰度图
            fn_gray_btn = gradio.Button("灰度图")
            # 按钮:转为二值图
            fn_binary_btn = gradio.Button("二值图")
            # 按钮:加入椒盐噪声
            fn_salt_and_pepper_noise_btn = gradio.Button("添加椒盐噪声")
            # 按钮:去除噪声
            fn_remove_noise_btn = gradio.Button("去除常规噪声")
            # 按钮:添加频域水印
            fn_watermark_btn = gradio.Button("添加频域水印")
            
        #绑定fn_gray_btn点击函数
        fn_gray_btn.click(fn=fn_gray, inputs=[before_img], outputs=after_img)
        # 绑定fn_binary_btn点击函数
        fn_binary_btn.click(fn=fn_binary, inputs=[before_img], outputs=after_img)
        # 绑定fn_salt_and_pepper_noise_btn点击函数
        fn_salt_and_pepper_noise_btn.click(fn=fn_salt_and_pepper_noise, inputs=[before_img], outputs=after_img)
        # 绑定fn_remove_noise_btn点击函数
        fn_remove_noise_btn.click(fn=fn_remove_noise,inputs=[before_img],outputs=after_img)
        # # 绑定
        fn_watermark_btn.click(fn=fn_watermark, inputs=[before_img], outputs=after_img)

    gradio.Markdown("## 视频处理")
    # 纵向排列
    with gradio.Column():
        # 横向排列
        with gradio.Row():
            before_video = gradio.Video(label="原视频")
            before_video_webcam = gradio.Image(sources="webcam", streaming=True,label="摄像头预览",visible=True)
            after_video = gradio.Video(label="处理后视频")
            shotcut_video = gradio.Image(label="视频截图")

        # 横向排列
        gradio.Markdown("视频操作区")
        with gradio.Row():
            # 按钮:视频截屏按钮
            fn_screenshot_btn = gradio.Button("视频截图")
            # 按钮:摄像头截图
            fn_screenshot_webcam_btn = gradio.Button("摄像头截图")
            # 按钮:摄像头视频保存3s
            fn_save_video_webcam_btn = gradio.Button("摄像头视频保存")
            # 按钮:截取第五帧
            fn_screenshot_frame_5_btn = gradio.Button("截图第五帧")
            # # 按钮:随机保存视频
            # fn_save_video_random_btn = gradio.Button("随机保存视频")
            # 按钮:视频倒放
            fn_video_upend_btn = gradio.Button("倒放")
            # 按钮:智能视频分析
            fn_open_analysis_btn = gradio.Button("视频智能分析")
            # 按钮:截图分析
            fn_shutcut_analysis_btn = gradio.Button("截图分析")
            # 按钮:人脸验证
            fn_face_comparison_btn = gradio.Button("人脸验证")

        # 横向排列
        with gradio.Row():
            gradio.Examples(examples=examples_videos, inputs=[before_video], label="示例视频")
            shotcut_analysis_video = gradio.Image(label="视频截图分析-图示")
            shotcut_analysis_text = gradio.Textbox(label="视频截图分析-文示", lines=4, placeholder="点击按钮开始分析...",)
            face_comparison_text = gradio.Textbox(label="人脸验证结果", lines=4, placeholder="点击按钮开始分析...",)
            
        with gradio.Row():
            video_smart_analysis_result_text = gradio.Textbox(label="视频智能分析结果", lines=4, placeholder="点击按钮开始分析...",)
            # notification_text = gradio.Textbox(label="预警提示", lines=4, placeholder="点击按钮开始分析...",)
        
        # 关键帧展示
        face_how = gradio.Image(label="人脸展示")
        with gradio.Row():
            gradio.Examples(examples=face_collection, inputs=[face_how], label="人脸集合")
            # keyframe_text = gradio.Textbox(label="预警提示", lines=4, placeholder="点击按钮开始分析...",)
        
        # 绑定按钮功能
        fn_open_analysis_btn.click(fn=fn_open_analysis,inputs=[before_video],outputs=[after_video,video_smart_analysis_result_text])
        fn_shutcut_analysis_btn.click(fn=fn_screenshot_analysis,inputs=[shotcut_video],outputs=[shotcut_analysis_video,shotcut_analysis_text])
        fn_face_comparison_btn.click(fn=fn_face_comparison,inputs=[shotcut_video],outputs=[face_comparison_text])

        fn_screenshot_btn.click(fn=fn_screenshot,inputs=[before_video],outputs=[shotcut_video,before_img])
        fn_screenshot_webcam_btn.click(fn=fn_screenshot_webcam,inputs=[before_video_webcam],outputs=[shotcut_video,before_img])
        fn_screenshot_frame_5_btn.click(fn=fn_screenshot_frame_5,inputs=[before_video],outputs=[shotcut_video,before_img])
        # fn_save_video_random_btn.click(fn=fn_save_video_random,inputs=[before_video],outputs=[after_video])
        fn_video_upend_btn.click(fn=fn_video_upend,inputs=[before_video],outputs=[after_video])
        fn_save_video_webcam_btn.click(fn=fn_save_video_webcam,inputs=[before_video_webcam],outputs=[after_video])

# 启动demo界面
demo.launch(share=True)

##################end 界面构建##################
