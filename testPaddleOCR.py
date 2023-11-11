from paddleocr import PaddleOCR
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math

lang = "korean" # 'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'

isPreprocessing = 1 # 0: 원본, 1: 전처리

filepath = "A.jpeg" # 파일 경로

font_path = "./NanumGothic.ttf" # 폰트 경로

# ----------------------------------------------

file_name, file_extension = os.path.splitext(filepath)

if not file_extension.startswith('.'):
    file_extension = '.' + file_extension

ocr = PaddleOCR(lang=lang, use_gpu=False)

img_path = f"./dolist/{file_name}{file_extension}"

def rotate_image_based_on_exif(image_path):
    img = Image.open(image_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass

    return img

rotated_img = rotate_image_based_on_exif(img_path)
rotated_img_path = "./rotated/rotated_" + f"{file_name}{file_extension}"
rotated_img.save(rotated_img_path)

image = cv2.imread(rotated_img_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

ret, thresh = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

preprocessed_img_path = f"./preprocessed/{file_name}_preprocessed{file_extension}"
cv2.imwrite(preprocessed_img_path, thresh)

if isPreprocessing:
    doPath = preprocessed_img_path
else:
    doPath = rotated_img_path

result = ocr.ocr(doPath, cls=True)

json_data = []

# print(result)

for line in result[0]:
    text, confidence = line[1]
    box = line[0]
    if confidence > 0.5:
        detection = {
            "text": text,
            "confidence": confidence,
            "box": box
        }
        json_data.append(detection)
json_data_sorted = sorted(json_data, key=lambda x: x['confidence'], reverse=True)

# print("-" * 50)
# print(json_data)
print("-" * 50)

def draw_detections(image_path, detections, output_path):
    image = Image.open(image_path)
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 128)
    for detection in detections:
        box = detection["box"]
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
    
img_path_out = f"./out/{file_name}-{lang}({isPreprocessing})-out.png"
draw_detections(rotated_img_path, json_data, img_path_out)
print(img_path_out)

with open(f'./result/ocr_result-{file_name}-{lang}({isPreprocessing}).json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

for detection in json_data_sorted:
    text = detection["text"]
    confidence = detection["confidence"]
    box = detection["box"]
    # print(text + "   : " + str(confidence))