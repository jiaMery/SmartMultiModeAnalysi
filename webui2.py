import cv2
import gradio
import random
import numpy as np
import os
import subprocess
import time
import shutil
from PIL import Image

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

def fn_open_operation(image, kernel_size=random.randint(1,100)):
    '''
    开运算函数
    参数：
    - image: 输入的图像
    - kernel_size: 开运算的核大小
    返回：
    - opened_image: 开运算后的图像
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def fn_close_operation(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    '''
    闭运算函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    返回：
    - closed_image: 闭运算后的图像
    '''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def fn_connected_domains(image: np.ndarray) -> np.ndarray:
    '''
    连通域函数
    参数：
    - image: 输入的二值图像
    返回：
    - labeled_image: 标记了连通域的图像
    - num_labels: 连通域的数量
    '''
    _, labeled_image = cv2.connectedComponents(image)
    num_labels = np.max(labeled_image)
    return labeled_image

def fn_erosion(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    '''
    腐蚀函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    返回：
    - eroded_image: 腐蚀后的图像
    '''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

def fn_dilation(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    '''
    膨胀函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    返回：
    - dilated_image: 膨胀后的图像
    '''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

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

def fn_gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
    '''
    高斯滤波函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    - sigma: 高斯核标准差,默认为0
    返回：
    - filtered_image: 滤波后的图像
    '''
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered_image

def fn_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    '''
    中值滤波函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    返回：
    - filtered_image: 滤波后的图像
    '''
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def fn_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    '''
    均值滤波函数
    参数：
    - image: 输入的图像
    - kernel_size: 卷积核大小,默认为3
    返回：
    - filtered_image: 滤波后的图像
    '''
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def fn_gray_mapping(image: np.ndarray) -> np.ndarray:
    '''
    灰度映射函数
    参数：
    - image: 输入的图像
    - mapping_func: 灰度映射函数
    返回：
    - mapped_image: 映射后的图像
    '''
    mapped_image = fn_binary(image)
    return mapped_image

def fn_image_enhancement(image: np.ndarray) -> np.ndarray:
    '''
    图像增强函数
    参数：
    - image: 输入的图像
    - enhancement_func: 图像增强函数
    返回：
    - enhanced_image: 增强后的图像
    '''
    #enhanced_image = enhancement_func(image)
    return image

def fn_walsh_transform(signal: np.ndarray) -> np.ndarray:
    '''
    Walsh变换函数
    参数：
    - signal: 输入的信号
    返回：
    - transformed_signal: 变换后的信号
    '''
    transformed_signal = np.fft.ifftshift(np.fft.fft(signal))
    return transformed_signal

def fn_hadamard_transform(signal: np.ndarray) -> np.ndarray:
    '''
    Hadamard变换函数
    参数：
    - signal: 输入的信号
    返回：
    - transformed_signal: 变换后的信号
    '''
    transformed_signal = np.fft.ifftshift(np.fft.fft(signal))
    return transformed_signal

def fn_1d_dct(signal: np.ndarray) -> np.ndarray:
    '''
    一维离散余弦变换函数
    参数：
    - signal: 输入的信号
    返回：
    - transformed_signal: 变换后的信号
    '''
    transformed_signal = np.fft.ifftshift(np.fft.fft(signal))
    return transformed_signal

def fn_2d_dct(image: np.ndarray) -> np.ndarray:
    '''
    二维离散余弦变换函数
    参数：
    - image: 输入的图像
    返回：
    - transformed_image: 变换后的图像
    '''
    transformed_image = cv2.dct(image)
    return transformed_image

def fn_2d_continuous_fourier_transform(image: np.ndarray) -> np.ndarray:
    '''
    二维连续傅里叶变换函数
    参数：
    - image: 输入的图像
    返回：
    - transformed_image: 变换后的图像
    '''
    transformed_image = np.fft.fft2(image)
    return transformed_image

def fn_2d_discrete_fourier_transform(image: np.ndarray) -> np.ndarray:
    '''
    二维离散傅里叶变换函数
    参数：
    - image: 输入的图像
    返回：
    - transformed_image: 变换后的图像
    '''
    transformed_image = np.fft.fftshift(np.fft.fft2(image))
    return transformed_image

def fn_continuous_wavelet_transform(signal: np.ndarray) -> np.ndarray:
    '''
    连续小波变换函数
    参数：
    - signal: 输入的信号
    - wavelet: 小波函数
    返回：
    - transformed_signal: 变换后的信号
    '''
    #transformed_signal = pywt.cwt(signal, wavelet)
    return signal

def fn_1d_discrete_wavelet_transform(signal: np.ndarray) -> np.ndarray:
    '''
    一维离散小波变换函数
    参数：
    - signal: 输入的信号
    - wavelet: 小波函数
    返回：
    - transformed_signal: 变换后的信号
    '''
    #transformed_signal = pywt.wavedec(signal, wavelet)
    return signal

def fn_2d_discrete_wavelet_transform(image: np.ndarray) -> np.ndarray:
    '''
    二维离散小波变换函数
    参数：
    - image: 输入的图像
    - wavelet: 小波函数
    返回：
    - transformed_image: 变换后的图像
    '''
    #transformed_image = pywt.wavedec2(image, wavelet)
    return image

def fn_image_skeletonization(image: np.ndarray) -> np.ndarray:
    '''
    图像骨架化函数
    参数：
    - image: 输入的二值图像
    返回：
    - skeletonized_image: 骨架化后的图像
    '''
    skeletonized_image = cv2.ximgproc.thinning(image)
    return skeletonized_image

# 打开摄像头并显示视频
def fn_open_webcam(is_open:bool=True):
    '''
    打开摄像头画面
    '''
    global before_video_webcam
    global before_video
    before_video_webcam.visible = True
    before_video.source="webcam"

# 关闭摄像头画面
def fn_close_webcam(is_open:bool=False):
    '''
    关闭摄像头画面
    '''
    global before_video_webcam
    global before_video
    before_video_webcam.visible=False

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

    return output_path

# 视频添加水印
def fn_add_watermark_video(video_path):
    '''
    给视频添加水印
    '''
    output_path = f'video_watermark_{random.randint(100,200)}.mp4'
    watermark_text = f'qsbye_{random.randint(100,200)}'
    shell_command = f'ffmpeg -i "{video_path}" -vf "drawtext=text=\'{watermark_text}\':x=w-tw-10:y=10:fontsize=24:fontcolor=white:shadowcolor=black:shadowx=2:shadowy=2" -c:a copy "{output_path}"'
    subprocess.call(shell_command, shell=True)
    return output_path

# 视频倒放
def fn_video_upend(video_path):
    '''
    视频倒放
    '''
    output_path = f'video_upend_{random.randint(100,200)}.mp4'
    shell_command = f'ffmpeg -i "{video_path}" -vf "reverse" -af "areverse" "{output_path}"'
    subprocess.call(shell_command, shell=True)
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
    
    return output_path


##################end 功能函数##################

##################start 界面构建##################

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

# 示例视频
examples_videos = [
    ["data/video/person.mp4"],
    ["data/video/bezos_vogels_contentVideo.mp4"],
]

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
            # 按钮:添加水印功能
            fn_add_watermark_video_btn = gradio.Button("添加水印")
            # 按钮:截取第五帧
            fn_screenshot_frame_5_btn = gradio.Button("截图第五帧")
            # 按钮:随机保存视频
            fn_save_video_random_btn = gradio.Button("随机保存视频")
            # 按钮:视频倒放
            fn_video_upend_btn = gradio.Button("倒放")

        # 横向排列
        with gradio.Row():
            gradio.Examples(examples=examples_videos, inputs=[before_video], label="示例视频")
            # 按钮:智能视频分析
            fn_open_analysis_btn = gradio.Button("视频智能分析")
            # 按钮:关闭浏览器摄像头按钮
            fn_key_frame_btn = gradio.Button("关键帧抽取")
            
        with gradio.Row():
            video_smart_analysis_result_text = gradio.Textbox(label="视频智能分析结果", lines=4, placeholder="点击按钮进行分析...",)
            notification_text = gradio.Textbox(label="预警提示", lines=4, placeholder="点击按钮进行分析...",)
        
        # 关键帧展示
        key_frame = gradio.Image(label="单帧关键帧展示")
        gradio.Examples(examples=examples_imgs, inputs=[key_frame], label="关键帧")
        
        # 绑定按钮功能
        # fn_open_analysis_btn.click(fn=fn_open_webcam,outputs=[before_video])
        # fn_key_frame_btn.click(fn=fn_close_webcam,outputs=[before_video])
        fn_screenshot_btn.click(fn=fn_screenshot,inputs=[before_video],outputs=[shotcut_video,before_img])
        fn_screenshot_webcam_btn.click(fn=fn_screenshot_webcam,inputs=[before_video_webcam],outputs=[shotcut_video,before_img])
        fn_screenshot_frame_5_btn.click(fn=fn_screenshot_frame_5,inputs=[before_video],outputs=[shotcut_video,before_img])
        fn_save_video_random_btn.click(fn=fn_save_video_random,inputs=[before_video],outputs=[after_video])
        fn_add_watermark_video_btn.click(fn=fn_add_watermark_video,inputs=[before_video],outputs=[after_video])
        fn_video_upend_btn.click(fn=fn_video_upend,inputs=[before_video],outputs=[after_video])
        fn_save_video_webcam_btn.click(fn=fn_save_video_webcam,inputs=[before_video_webcam],outputs=[after_video])

# 启动demo界面
demo.launch(share=True)

##################end 界面构建##################
