import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from PIL import Image
import json
import os
import numpy as np
from util import *

face_detector = pipeline(Tasks.face_detection, model='gaosheng/face_detect')
# face_recognizer = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
face_recognizer = pipeline(Tasks.face_recognition, model='iic/cv_ir101_facerecognition_cfglint')
face_bank = load_face_bank('face_bank/', face_recognizer)

def inference(img: Image, draw_detect_enabled, detect_threshold, sim_threshold) -> json:
    img = resize_img(img)
    img = img.convert('RGB')
    detection_result = face_detector(img)

    boxes = np.array(detection_result[OutputKeys.BOXES])
    scores = np.array(detection_result[OutputKeys.SCORES])
    faces = []

    for i in range(len(boxes)):
        score = scores[i]
        if score < detect_threshold:
            continue
        box = boxes[i]
        face_embedding = get_face_embedding(img, box, face_recognizer)
        name, sim = get_name_sim(face_embedding, face_bank)
        if name is None:
            continue
        if sim < sim_threshold:
            faces.append({'box': box, 'name': '未知', 'sim': sim})
        else:
            faces.append({'box': box, 'name': name, 'sim': sim})
    rows = get_rows(faces)
    row_names = get_row_names(faces, rows)
    draw_name(img, row_names)
    if draw_detect_enabled:
        draw_faces(img, faces)
    return img, get_row_names_text(row_names)

examples = ['example.jpg']

with gr.Blocks() as demo:
    with gr.Row():
        draw_detect_enabled = gr.Checkbox(label="是否画框", value=True)
        detect_threshold = gr.Slider(label="检测阈值", minimum=0, maximum=1, value=0.3)
        sim_threshold = gr.Slider(label="识别阈值", minimum=0, maximum=1, value=0.3)
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", height=350)
            submit = gr.Button("提交")
        with gr.Column():
            img_output = gr.Image(type="pil")
            name_output = gr.Text(label="人名")
    submit.click(
        fn=inference, 
        inputs=[img_input, draw_detect_enabled, detect_threshold, sim_threshold], 
        outputs=[img_output, name_output])
    gr.Examples(examples, inputs=[img_input])

# demo.launch()