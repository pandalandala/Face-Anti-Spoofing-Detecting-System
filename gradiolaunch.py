# import os
# import subprocess
# import time
#
# import cv2
# import gradio as gr
# from PIL import Image
# import numpy as np
# from inference import inference
#
#
# def detect(image):
#
#     save_path = "detlandmark/inference_images/0.jpg"
#     cv2.imwrite(save_path, image)
#
#     command = "python inference.py inference --images='detlandmark/inference_images/*.jpg'"
#     # 使用 subprocess 启动新的命令行并运行命令
#     process = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
#     # 等待命令行命令执行完成
#     process.wait()
#
#     with open('result/inference.txt', 'r') as file:
#         output = file.read()
#     return output
#
# inputs = gr.Image()
# gr.Radio(["MyresNet34", "MyVggNet11"])
# outputs = "text"

# interface = gr.Interface(fn=detect,
#                          inputs=inputs,
#                          outputs=outputs,
#                          # live=True,
#                          title="FAS Detection System")
#
# img_path = "detlandmark/inference_images/"
# for filename in os.listdir(img_path):
#     file_path = os.path.join(img_path, filename)
#     # 如果是文件，则直接删除
#     if os.path.isfile(file_path):
#         os.remove(file_path)
#         print(f"已删除文件: {file_path}")
#
# interface.launch()
#
import gradio as gr
import torch
from torchvision import transforms
import cv2
import numpy as np
import requests
from PIL import Image

# 加载ResNet模型
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict(img):
    # 将图像转换为OpenCV格式
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 在图像中检测人脸
    faces = face_cascade.detectMultiScale(img_cv2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 如果检测到人脸
    if len(faces) > 0:
        # 取第一个人脸
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

        # 预处理人脸图像
        face_pil = Image.fromarray(face_img)
        input_tensor = preprocess(face_pil).unsqueeze(0)

        # 使用模型进行预测
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)

        # 确定真假标签
        # 这里我们假设模型对真假的预测结果是 0 表示假（fake），1 表示真（real）
        # 如果模型输出的概率大于等于0.5，则认为是真实的（real），否则认为是伪造的（fake）
        label = 1 if prediction[1] >= 0.5 else 0
        probability = prediction[label].item()
    else:
        label = "No face detected"
        probability = 0.0

    return label, probability

# 输入和输出界面
iface = gr.Interface(fn=predict, inputs="image", outputs=["label", "number"])
iface.launch()