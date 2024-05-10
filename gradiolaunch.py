import os
import subprocess
import time

import cv2
import gradio as gr
from PIL import Image
import numpy as np
from inference import inference


def detect(image):

    save_path = "detlandmark/inference_images/0.jpg"
    cv2.imwrite(save_path, image)

    command = "python inference.py inference --images='detlandmark/inference_images/*.jpg'"
    # 使用 subprocess 启动新的命令行并运行命令
    subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    # 等待命令行命令执行完成
    subprocess.wait()

    with open('result/inference.txt', 'r') as file:
        output = file.read()
    return output


interface = gr.Interface(detect,
                         inputs=[
                             "image",
                             # gr.Radio(["MyresNet34", "MyVggNet11"])
                         ],
                         outputs="text",
                         live=True,
                         title="FAS Detection System")

img_path = "detlandmark/inference_images/"
for filename in os.listdir(img_path):
    file_path = os.path.join(img_path, filename)
    # 如果是文件，则直接删除
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"已删除文件: {file_path}")

interface.launch()