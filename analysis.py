import time
start = time.time()
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math
import re

filepath = "A.jpeg"

file_name, file_extension = os.path.splitext(filepath)
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

results = [result_en0, result_en1, result_ko0, result_ko1, result_enko0, result_enko1]

def standardize_date(date_str):
    # 정규 표현식을 사용하여 날짜 패턴 찾기
    match = re.search(r'(\d{4})?\.?(\d{1,2})\.(\d{1,2})', date_str)
    if match:
        # 연도가 있는 경우와 없는 경우를 처리
        year = match.group(1) if match.group(1) else '2023'  # 연도가 없으면 2023으로 가정
        month = match.group(2).zfill(2)  # 월을 2자리로 맞춤
        day = match.group(3).zfill(2)    # 일을 2자리로 맞춤
        return f"{year}.{month}.{day}"
    return None

def process_text(text):
    # '~' 기호를 포함하는 경우, 텍스트를 두 부분으로 나눔
    if '~' in text:
        start_text, end_text = text.split('~', 1)
        start_date = standardize_date(start_text.strip())

        # 두 번째 텍스트에서 연도가 누락된 경우, 첫 번째 텍스트의 연도를 사용
        if re.search(r'\d{4}', end_text):
            end_date = standardize_date(end_text.strip())
        else:
            year = start_date.split('.')[0] if start_date else '2023'  # 연도가 없으면 2023으로 가정
            end_date = standardize_date(year + '.' + end_text.strip())

        return f"{start_date} ~ {end_date}" if start_date and end_date else None
    else:
        return standardize_date(text.strip())

# 모든 가능한 날짜 패턴을 생성
date_patterns = ['\\d{1,2}\\.Mar(?:ch)?\\.\\d{4}', 'Aug(?:ust)?-\\d{1,2}-\\d{2}', 'Feb(?:ruary)?/\\d{1,2}/\\d{2}', '\\d{4}/Dec(?:ember)?/\\d{1,2}', '\\d{4}-May-\\d{1,2}', '\\d{4}/Jul(?:y)?/\\d{1,2}', '\\d{1,2}/Sep(?:tember)?/\\d{4}', '\\d{4}-\\d{1,2}-\\d{1,2}', '\\d{4}\\sSep(?:tember)?\\s\\d{1,2}', '\\d{4}\\.Dec(?:ember)?\\.\\d{1,2}', '\\d{1,2}/Jan(?:uary)?/\\d{2}', '\\d{2}\\.Jul(?:y)?\\.\\d{1,2}', '\\d{2}/Aug(?:ust)?/\\d{1,2}', '\\d{4}\\.Apr(?:il)?\\.\\d{1,2}', '\\d{1,2}/Feb(?:ruary)?/\\d{4}', '\\d{4}\\s\\d{1,2}\\s\\d{1,2}', '\\d{2}/Feb(?:ruary)?/\\d{1,2}', '\\d{1,2}\\.Apr(?:il)?\\.\\d{2}', '\\d{1,2}\\.Jul(?:y)?\\.\\d{2}', 'Dec(?:ember)?\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}/Oct(?:ober)?/\\d{4}', '\\d{1,2}/Dec(?:ember)?/\\d{4}', '\\d{1,2}-Aug(?:ust)?-\\d{4}', '\\d{1,2}\\sJul(?:y)?\\s\\d{2}', '\\d{1,2}\\sJul(?:y)?\\s\\d{4}', '\\d{1,2}-Feb(?:ruary)?-\\d{2}', '\\d{1,2}-Apr(?:il)?-\\d{4}', '\\d{2}\\sMay\\s\\d{1,2}', '\\d{1,2}-Jul(?:y)?-\\d{2}', '\\d{4}\\.Mar(?:ch)?\\.\\d{1,2}', '\\d{2}\\.Feb(?:ruary)?\\.\\d{1,2}', '\\d{1,2}/Aug(?:ust)?/\\d{2}', '\\d{1,2}-Sep(?:tember)?-\\d{4}', '\\d{4}/Sep(?:tember)?/\\d{1,2}', '\\d{1,2}\\sDec(?:ember)?\\s\\d{2}', '\\d{4}/\\d{1,2}/\\d{1,2}', '\\d{4}-Aug(?:ust)?-\\d{1,2}', '\\d{1,2}\\sFeb(?:ruary)?\\s\\d{2}', '\\d{2}-May-\\d{1,2}', '\\d{1,2}-Nov(?:ember)?-\\d{2}', '\\d{4}\\sJun(?:e)?\\s\\d{1,2}', '\\d{1,2}\\.\\d{1,2}\\.\\d{4}', '\\d{2}/Jan(?:uary)?/\\d{1,2}', '\\d{2}-Jan(?:uary)?-\\d{1,2}', '\\d{2}\\sApr(?:il)?\\s\\d{1,2}', 'Apr(?:il)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}\\sNov(?:ember)?\\s\\d{2}', '\\d{2}\\.Oct(?:ober)?\\.\\d{1,2}', '\\d{4}/Jan(?:uary)?/\\d{1,2}', '\\d{1,2}/\\d{1,2}/\\d{4}', '\\d{1,2}-Feb(?:ruary)?-\\d{4}', '\\d{1,2}\\sMay\\s\\d{4}', '\\d{2}\\s\\d{1,2}\\s\\d{1,2}', '\\d{4}\\.Feb(?:ruary)?\\.\\d{1,2}', '\\d{1,2}-Dec(?:ember)?-\\d{4}', '\\d{4}\\.Jun(?:e)?\\.\\d{1,2}', '\\d{1,2}-Sep(?:tember)?-\\d{2}', '\\d{1,2}\\sDec(?:ember)?\\s\\d{4}', '\\d{2}/Dec(?:ember)?/\\d{1,2}', '\\d{2}\\.Nov(?:ember)?\\.\\d{1,2}', 'Jun(?:e)?\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}-Oct(?:ober)?-\\d{4}', '\\d{2}\\.Apr(?:il)?\\.\\d{1,2}', '\\d{4}/Oct(?:ober)?/\\d{1,2}', 'Oct(?:ober)?\\.\\d{1,2}\\.\\d{2}', '\\d{4}/Feb(?:ruary)?/\\d{1,2}', '\\d{4}\\.Nov(?:ember)?\\.\\d{1,2}', '\\d{2}\\sMar(?:ch)?\\s\\d{1,2}', '\\d{1,2}/Jan(?:uary)?/\\d{4}', '\\d{1,2}\\sJan(?:uary)?\\s\\d{4}', '\\d{4}\\.Jul(?:y)?\\.\\d{1,2}', '\\d{2}\\.Jun(?:e)?\\.\\d{1,2}', '\\d{2}-Dec(?:ember)?-\\d{1,2}', '\\d{4}\\sMar(?:ch)?\\s\\d{1,2}', 'Apr(?:il)?\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}\\sJan(?:uary)?\\s\\d{2}', '\\d{2}-Jun(?:e)?-\\d{1,2}', 'Sep(?:tember)?\\s\\d{1,2}\\s\\d{2}', 'Nov(?:ember)?/\\d{1,2}/\\d{2}', '\\d{1,2}-Mar(?:ch)?-\\d{4}', '\\d{4}\\sOct(?:ober)?\\s\\d{1,2}', '\\d{1,2}\\sSep(?:tember)?\\s\\d{4}', 'Jan(?:uary)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}\\.Jan(?:uary)?\\.\\d{4}', '\\d{2}-Sep(?:tember)?-\\d{1,2}', '\\d{1,2}\\.Sep(?:tember)?\\.\\d{4}', '\\d{1,2}/Jun(?:e)?/\\d{2}', '\\d{1,2}\\.Aug(?:ust)?\\.\\d{4}', '\\d{1,2}\\sNov(?:ember)?\\s\\d{4}', '\\d{4}-Sep(?:tember)?-\\d{1,2}', 'Jan(?:uary)?-\\d{1,2}-\\d{2}', '\\d{1,2}\\.\\d{1,2}\\.\\d{2}', '\\d{2}\\sFeb(?:ruary)?\\s\\d{1,2}', '\\d{4}-Mar(?:ch)?-\\d{1,2}', '\\d{1,2}\\sAug(?:ust)?\\s\\d{2}', '\\d{1,2}/Apr(?:il)?/\\d{4}', '\\d{2}\\.May\\.\\d{1,2}', 'Sep(?:tember)?-\\d{1,2}-\\d{2}', '\\d{1,2}/Oct(?:ober)?/\\d{2}', '\\d{1,2}-Jun(?:e)?-\\d{2}', '\\d{2}\\sJan(?:uary)?\\s\\d{1,2}', '\\d{1,2}/Dec(?:ember)?/\\d{2}', '\\d{1,2}\\.Mar(?:ch)?\\.\\d{2}', 'Aug(?:ust)?/\\d{1,2}/\\d{2}', 'Jun(?:e)?/\\d{1,2}/\\d{2}', 'Jul(?:y)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}\\sApr(?:il)?\\s\\d{4}', '\\d{1,2}/Mar(?:ch)?/\\d{4}', 'Feb(?:ruary)?\\.\\d{1,2}\\.\\d{2}', '\\d{4}/May/\\d{1,2}', 'Aug(?:ust)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}-\\d{1,2}-\\d{2}', '\\d{2}-Mar(?:ch)?-\\d{1,2}', '\\d{4}\\.Aug(?:ust)?\\.\\d{1,2}', '\\d{1,2}\\.May\\.\\d{4}', '\\d{2}\\sNov(?:ember)?\\s\\d{1,2}', '\\d{1,2}\\sApr(?:il)?\\s\\d{2}', '\\d{4}\\.Sep(?:tember)?\\.\\d{1,2}', 'Feb(?:ruary)?\\s\\d{1,2}\\s\\d{2}', '\\d{4}\\sMay\\s\\d{1,2}', '\\d{1,2}-Aug(?:ust)?-\\d{2}', '\\d{1,2}\\sOct(?:ober)?\\s\\d{4}', 'Nov(?:ember)?\\s\\d{1,2}\\s\\d{2}', 'Dec(?:ember)?-\\d{1,2}-\\d{2}', '\\d{4}\\.Jan(?:uary)?\\.\\d{1,2}', '\\d{2}\\.Sep(?:tember)?\\.\\d{1,2}', '\\d{1,2}-Mar(?:ch)?-\\d{2}', 'Jan(?:uary)?/\\d{1,2}/\\d{2}', '\\d{2}\\sOct(?:ober)?\\s\\d{1,2}', '\\d{1,2}\\.Jun(?:e)?\\.\\d{2}', 'May-\\d{1,2}-\\d{2}', '\\d{2}\\sDec(?:ember)?\\s\\d{1,2}', 'Apr(?:il)?/\\d{1,2}/\\d{2}', 'Jul(?:y)?\\s\\d{1,2}\\s\\d{2}', 'Aug(?:ust)?\\s\\d{1,2}\\s\\d{2}', '\\d{4}-Feb(?:ruary)?-\\d{1,2}', '\\d{1,2}\\.Feb(?:ruary)?\\.\\d{2}', '\\d{1,2}\\s\\d{1,2}\\s\\d{4}', '\\d{1,2}\\.Dec(?:ember)?\\.\\d{4}', '\\d{1,2}-Jan(?:uary)?-\\d{4}', 'Mar(?:ch)?/\\d{1,2}/\\d{2}', 'May\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}/May/\\d{4}', '\\d{1,2}\\sOct(?:ober)?\\s\\d{2}', 'Nov(?:ember)?\\.\\d{1,2}\\.\\d{2}', '\\d{2}/Apr(?:il)?/\\d{1,2}', '\\d{1,2}-Dec(?:ember)?-\\d{2}', '\\d{2}/Jun(?:e)?/\\d{1,2}', '\\d{1,2}\\sMar(?:ch)?\\s\\d{4}', '\\d{1,2}\\.Sep(?:tember)?\\.\\d{2}', '\\d{1,2}\\.Nov(?:ember)?\\.\\d{2}', '\\d{1,2}-Jun(?:e)?-\\d{4}', '\\d{1,2}\\sJun(?:e)?\\s\\d{4}', '\\d{1,2}\\.Jun(?:e)?\\.\\d{4}', '\\d{4}\\sAug(?:ust)?\\s\\d{1,2}', '\\d{4}/Nov(?:ember)?/\\d{1,2}', '\\d{4}-Jun(?:e)?-\\d{1,2}', 'Jun(?:e)?\\.\\d{1,2}\\.\\d{2}', '\\d{4}\\.\\d{1,2}\\.\\d{1,2}', '\\d{2}-\\d{1,2}-\\d{1,2}', '\\d{4}-Jul(?:y)?-\\d{1,2}', '\\d{1,2}-May-\\d{2}', '\\d{4}-Apr(?:il)?-\\d{1,2}', 'Oct(?:ober)?\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}\\.Jan(?:uary)?\\.\\d{2}', '\\d{1,2}\\sMar(?:ch)?\\s\\d{2}', '\\d{1,2}/Feb(?:ruary)?/\\d{2}', '\\d{2}\\.Jan(?:uary)?\\.\\d{1,2}', 'Jun(?:e)?-\\d{1,2}-\\d{2}', 'Oct(?:ober)?/\\d{1,2}/\\d{2}', '\\d{1,2}/Nov(?:ember)?/\\d{2}', '\\d{1,2}\\.Apr(?:il)?\\.\\d{4}', '\\d{1,2}-Jan(?:uary)?-\\d{2}', '\\d{2}/Mar(?:ch)?/\\d{1,2}', '\\d{4}\\sFeb(?:ruary)?\\s\\d{1,2}', '\\d{4}-Oct(?:ober)?-\\d{1,2}', '\\d{4}-Nov(?:ember)?-\\d{1,2}', '\\d{2}\\sSep(?:tember)?\\s\\d{1,2}', '\\d{1,2}-May-\\d{4}', '\\d{1,2}-Apr(?:il)?-\\d{2}', '\\d{1,2}/Jul(?:y)?/\\d{4}', '\\d{4}/Jun(?:e)?/\\d{1,2}', '\\d{4}/Aug(?:ust)?/\\d{1,2}', '\\d{2}-Jul(?:y)?-\\d{1,2}', '\\d{2}\\.Dec(?:ember)?\\.\\d{1,2}', '\\d{1,2}-Oct(?:ober)?-\\d{2}', 'May/\\d{1,2}/\\d{2}', '\\d{1,2}\\sAug(?:ust)?\\s\\d{4}', '\\d{1,2}/Nov(?:ember)?/\\d{4}', 'Jan(?:uary)?\\s\\d{1,2}\\s\\d{2}', '\\d{1,2}/May/\\d{2}', 'May\\s\\d{1,2}\\s\\d{2}', 'Sep(?:tember)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}\\sMay\\s\\d{2}', '\\d{1,2}/Sep(?:tember)?/\\d{2}', '\\d{1,2}\\.Feb(?:ruary)?\\.\\d{4}', '\\d{2}-Apr(?:il)?-\\d{1,2}', '\\d{4}-Dec(?:ember)?-\\d{1,2}', '\\d{2}/\\d{1,2}/\\d{1,2}', '\\d{1,2}-\\d{1,2}-\\d{4}', '\\d{4}\\sDec(?:ember)?\\s\\d{1,2}', 'Nov(?:ember)?-\\d{1,2}-\\d{2}', '\\d{2}-Aug(?:ust)?-\\d{1,2}', 'Mar(?:ch)?\\s\\d{1,2}\\s\\d{2}', '\\d{2}\\.Aug(?:ust)?\\.\\d{1,2}', '\\d{1,2}\\.Oct(?:ober)?\\.\\d{2}', '\\d{2}-Oct(?:ober)?-\\d{1,2}', '\\d{2}-Feb(?:ruary)?-\\d{1,2}', 'Sep(?:tember)?/\\d{1,2}/\\d{2}', '\\d{2}/Sep(?:tember)?/\\d{1,2}', '\\d{1,2}/Jul(?:y)?/\\d{2}', '\\d{1,2}\\.Nov(?:ember)?\\.\\d{4}', 'Mar(?:ch)?\\.\\d{1,2}\\.\\d{2}', '\\d{4}-Jan(?:uary)?-\\d{1,2}', '\\d{4}\\sJan(?:uary)?\\s\\d{1,2}', '\\d{1,2}\\sJun(?:e)?\\s\\d{2}', '\\d{2}\\.Mar(?:ch)?\\.\\d{1,2}', '\\d{1,2}/Jun(?:e)?/\\d{4}', 'Oct(?:ober)?-\\d{1,2}-\\d{2}', '\\d{1,2}\\sFeb(?:ruary)?\\s\\d{4}', '\\d{4}\\sJul(?:y)?\\s\\d{1,2}', '\\d{1,2}\\.Jul(?:y)?\\.\\d{4}', '\\d{2}\\sJun(?:e)?\\s\\d{1,2}', 'Dec(?:ember)?/\\d{1,2}/\\d{2}', '\\d{1,2}/\\d{1,2}/\\d{2}', '\\d{4}\\sNov(?:ember)?\\s\\d{1,2}', '\\d{4}\\.May\\.\\d{1,2}', 'Apr(?:il)?-\\d{1,2}-\\d{2}', '\\d{4}\\.Oct(?:ober)?\\.\\d{1,2}', '\\d{1,2}/Aug(?:ust)?/\\d{4}', '\\d{2}/Jul(?:y)?/\\d{1,2}', '\\d{2}/Oct(?:ober)?/\\d{1,2}', '\\d{2}\\sJul(?:y)?\\s\\d{1,2}', '\\d{2}/May/\\d{1,2}', '\\d{1,2}\\.May\\.\\d{2}', '\\d{1,2}\\sSep(?:tember)?\\s\\d{2}', '\\d{2}\\.\\d{1,2}\\.\\d{1,2}', '\\d{4}/Mar(?:ch)?/\\d{1,2}', '\\d{4}\\sApr(?:il)?\\s\\d{1,2}', '\\d{1,2}\\.Oct(?:ober)?\\.\\d{4}', '\\d{1,2}\\.Aug(?:ust)?\\.\\d{2}', '\\d{1,2}-Jul(?:y)?-\\d{4}', '\\d{2}/Nov(?:ember)?/\\d{1,2}', 'Jul(?:y)?-\\d{1,2}-\\d{2}', '\\d{1,2}\\.Dec(?:ember)?\\.\\d{2}', 'Jul(?:y)?/\\d{1,2}/\\d{2}', 'Mar(?:ch)?-\\d{1,2}-\\d{2}', '\\d{2}\\sAug(?:ust)?\\s\\d{1,2}', 'Feb(?:ruary)?-\\d{1,2}-\\d{2}', 'Dec(?:ember)?\\.\\d{1,2}\\.\\d{2}', '\\d{1,2}/Apr(?:il)?/\\d{2}', '\\d{4}/Apr(?:il)?/\\d{1,2}', '\\d{1,2}-Nov(?:ember)?-\\d{4}', '\\d{2}-Nov(?:ember)?-\\d{1,2}', '\\d{1,2}/Mar(?:ch)?/\\d{2}']


for result in results:
    for detection in result:
        box = detection['box']
        text = detection['text']
        confidence = detection['confidence']
        
        for pattern in date_patterns:
            if re.search(pattern, text):
                processed_text = process_text(text)
                if processed_text:
                    print(f"일정 발견 : {text}\n  =>  {processed_text}")
                    break


runtime = time.time() - start
print("Runtime : " + str(runtime) + "s")