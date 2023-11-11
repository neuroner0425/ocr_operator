import time
start = time.time()
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math

filepaths = ["A.jpeg","B.jpeg","C.png","D.jpeg","E.png","F.png","G.png","H.png","I.png","J.png"]

def draw_detections(image_path, detections, output_path):
    font_path = "./NanumGothic.ttf"
    image = Image.open(image_path)
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        box = detection["box"]
        confidence = detection["confidence"]
        outline_color = (0, 255, 0, 128)
        if confidence < 0.6:
            outline_color = (0, 0, 0, 128)
        elif confidence < 0.7:
            outline_color = (50, 50, 0, 128)
        elif confidence < 0.8:
            outline_color = (255, 0, 0, 128)
        elif confidence < 0.85:
            outline_color = (255, 50, 0, 128)
        elif confidence < 0.9:
            outline_color = (100, 180, 0, 128)
        flattened_box = [coordinate for point in box for coordinate in point]
        draw.polygon(flattened_box, outline=outline_color, fill=outline_color)

    combined = Image.alpha_composite(image.convert('RGBA'), overlay)
    combined = combined.convert('RGB')
    combined.save(output_path)
    
    image = Image.open(output_path)
    draw = ImageDraw.Draw(image)
    
    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    
    for detection in detections:
        box = detection["box"]
        text = f"{detection['text']} ({detection['confidence']:0.3f})"
        
        font_size = int((box[3][1] - box[0][1])*0.4)
        font = ImageFont.truetype(font_path, font_size)

        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)
        text_position = (min_x, min_y-font_size*0.5)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                draw.text((text_position[0] + dx, text_position[1] + dy), text, font=font, fill=outline_color)
        draw.text(text_position, text, font=font, fill=text_color)
    image.save(output_path)
    print("Saved to " + output_path)
    

def draw(_filepath):
    file_name, file_extension = os.path.splitext(_filepath)
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    result_en0 = []
    with open(f'./result/ocr_result-{file_name}-en(0).json', 'r') as file:
        result_en0 = json.load(file)
    result_en1 = []
    with open(f'./result/ocr_result-{file_name}-en(1).json', 'r') as file:
        result_en1 = json.load(file)
    result_ko0 = []
    with open(f'./result/ocr_result-{file_name}-korean(0).json', 'r') as file:
        result_ko0 = json.load(file)
    result_ko1 = []
    with open(f'./result/ocr_result-{file_name}-korean(1).json', 'r') as file:
        result_ko1 = json.load(file)
    result_enko0 = []
    with open(f'./result/ocr_result-{file_name}-enko(0).json', 'r') as file:
        result_enko0 = json.load(file)
    result_enko1 = []
    with open(f'./result/ocr_result-{file_name}-enko(1).json', 'r') as file:
        result_enko1 = json.load(file)
        
    img_path = "./rotated/rotated_" + f"{file_name}{file_extension}"
    img_path_out = f"./analysis/temp/{file_name}-en(0)-out.png"
    draw_detections(img_path, result_en0, img_path_out)
    img_path_out = f"./analysis/temp/{file_name}-en(1)-out.png"
    draw_detections(img_path, result_en1, img_path_out)
    img_path_out = f"./analysis/temp/{file_name}-korean(0)-out.png"
    draw_detections(img_path, result_ko0, img_path_out)
    img_path_out = f"./analysis/temp/{file_name}-korean(1)-out.png"
    draw_detections(img_path, result_ko1, img_path_out)
    img_path_out = f"./analysis/temp/{file_name}-enko(0)-out.png"
    draw_detections(img_path, result_enko0, img_path_out)
    img_path_out = f"./analysis/temp/{file_name}-enko(1)-out.png"
    draw_detections(img_path, result_enko1, img_path_out)

for filepath in filepaths:
    draw(filepath)

runtime = time.time() - start
print("Runtime : " + str(runtime) + "s")