import os
from PIL import Image,ImageDraw,ImageFont
import pickle
from sklearn.cluster import DBSCAN
import numpy as np
import glob
import speech_recognition as sr 

def load_face_bank(face_folder, face_recognizer, use_cache=True):
    cache_path = 'facebank.cache'
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    bank = []
    img_paths = glob.glob(os.path.join(face_folder, '**/**.**'))
    # 遍历已知人脸图片
    for img_path in img_paths:
        dirStr, ext = os.path.splitext(img_path)
        if ext not in ['.jpg', '.jpeg', '.png']:
            continue
        # 获取人名
        name = dirStr.split(os.sep)[-1]
        img = Image.open(img_path)
        img = img.convert('RGB')
        embeddings = face_recognizer(img)['img_embedding']
        if len(embeddings) == 0:
            continue
        bank.append({
            "name": name,
            "embedding": embeddings[0]
        })
    # 缓存特征库
    with open(cache_path, 'wb') as f:
        pickle.dump(bank, f)
    return bank

def get_face_img(image, box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    x0 = box[0] - w//2
    if x0 < 0:
        x0 = 1
    y0 = box[1] - h//2
    if y0 < 0:
        y0 = 1
    x1 = box[2] + w//2
    if x1 > image.width:
        x1 = image.width - 1
    y1 = box[3] + h//2
    if y1 > image.height:
        y1 = image.height - 1
    return image.crop((x0, y0, x1, y1))

def get_face_embedding(image, box, face_recognizer):
    face_img = get_face_img(image, box)
    embeddings = face_recognizer(face_img)['img_embedding']
    if len(embeddings) == 0:
        return None
    return embeddings[0]

def get_name_sim(face_embedding, face_bank):
    name = ''
    maxSim = 0
    for face in face_bank:
        sim = np.dot(face_embedding, face['embedding'])
        if sim > maxSim:
            maxSim = sim
            name = face['name']
    return name, maxSim

def get_emotion(img, emotion_recognizer):
    ret = emotion_recognizer(img)

    if not ret['labels'] or not ret['scores']:
        return 'unknow'

    label_idx = np.array(ret['scores']).argmax()
    label = ret['labels'][label_idx]
    return label

def draw_face(img_draw, face, font):
    box = face['box']
    name = face['name']
    sim = face['sim']

    img_draw.text((box[0]-10, box[1]-20), name, fill=(0, 0, 255), font=font)
    img_draw.rectangle([box[0], box[1], box[2], box[3]], outline ='red')

def draw_emotion(image, img_draw, face, font, emotion_recognizer):
    box = face['box']
    face_img = get_face_img(image, box)
    emotion = get_emotion(face_img, emotion_recognizer)
    img_draw.text((box[0]-5, box[1]-35), emotion, fill=(0, 255, 0), font=font)

def draw_faces(image, faces, emotion_recognizer):
    font = ImageFont.truetype("Microsoft YaHei UI Bold.ttf", 15, encoding="unic")
    draw = ImageDraw.Draw(image)
    for face in faces:
        draw_face(draw, face, font)
        draw_emotion(image, draw, face, font, emotion_recognizer)

def get_rows(faces):
    # 获取人脸检测框高度的平均值，作为DBSCAN算法的eps参数
    boxes = [face['box'] for face in faces]
    mean_h = get_mean_height(boxes)
    ys = [(box[1] + box[3])//2 for box in boxes]
    # 使用y坐标作为距离度量值
    data = np.expand_dims(np.array(ys), axis=1)
    dbscan = DBSCAN(eps=mean_h*0.395, min_samples=5)
    # 获取到每个度量值对应的类别
    labels = dbscan.fit_predict(data)
    rows = []
    # 聚类出来的类别数即为排数
    for i in range(max(labels)+1):
        columns = []
        top = 0
        for j in range(len(boxes)):
            if i == labels[j]:
                # 加入对应排
                columns.append((boxes[j][0], j))
                top += boxes[j][1]
        # 排内按照x坐标排序
        columns.sort(key=lambda x: x[0])
        rows.append((top // len(columns), [item[1] for item in columns]))
        # 排按照y坐标排序
        rows.sort(key=lambda x: x[0])
    return [row[1] for row in rows]

def draw_name(img, row_names):
    line_space = 10
    bottom_shift = 50
    # 使用中文字体
    font = ImageFont.truetype("Microsoft YaHei UI Bold.ttf", 10, encoding="unic")
    draw = ImageDraw.Draw(img)
    height_count = 0
    for row_name in row_names:
        y = img.height - bottom_shift + height_count * line_space
        name_str = ''
        for name in row_name:
            name_str += f'{name}   '
        name_str = name_str.strip()
        # 计算人名字符串渲染到图片中所占的长度
        text_len = draw.textlength(name_str, font)
        x = (img.width - text_len) //2
        # 将人名字符串居中渲染到图片中
        draw.text((x, y), name_str, fill=(0, 128, 0), font=font)
        height_count += 1
    return img

def get_row_names(faces, rows):
    row_names = []
    for row in rows:
        row_name = []
        for index in row:
            row_name.append(faces[index]['name'])
        row_names.append(row_name)
    return row_names

def get_row_names_text(row_names):
    text = ''
    for row_name in row_names:
        for name in row_name:
            text += f'{name}   '
        text += '\n'
    return text.rstrip('\n')

def get_mean_height(boxes):
    h_sum = 0
    for box in boxes:
        h_sum += box[3] - box[1]
    return h_sum // len(boxes)

def resize_img(img):
    ratio = img.width / 1000
    if ratio < 1:
        return img
    w_new = int(img.width / ratio)
    h_new = int(img.height / ratio)
    return img.resize((w_new, h_new), resample=Image.BILINEAR)

def speech_to_text(audio):  
    # 实例化语音识别器  
    recognizer = sr.Recognizer()  

    # 将音频数据转换为AudioFile对象  
    with sr.AudioFile(audio) as source:  
        audio_data = recognizer.record(source)  
          
    try:  
        # 使用Google Web Speech API将音频转换为文本  
        text = recognizer.recognize_google(audio_data, language="zh-CN")  
    except sr.UnknownValueError:  
        text = "无法识别音频"  
    except sr.RequestError:  
        text = "语音识别服务不可用"  

    return text
