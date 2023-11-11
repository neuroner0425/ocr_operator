from PIL import Image

# 이미지 파일을 엽니다
image = Image.open('ocr_test-2.png')

# 이미지를 'L' 모드로 변환하여 흑백으로 만듭니다
bw_image = image.convert('L')

# 변환된 이미지를 저장합니다
bw_image.save('ocr_test.png')
