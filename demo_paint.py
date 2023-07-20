#!/usr/bin/env python
# encoding: utf-8
import gradio as gr

from VisCPM import VisCPMPaint

# 修改你的模型地址
model_path = 'path to checkpoint'
painter = VisCPMPaint(model_path, image_safety_checker=False, prompt_safety_checker=False, add_ranker=True)
print("load  image model  success !")


def gen_img(txt, imgs):
    image = painter.generate(txt)
    imgs.append(image)
    return "",imgs,imgs


with gr.Blocks() as demo:
    imgs = gr.State([])
    gallery = gr.Gallery(label="生成图片")
    txt_message = gr.Textbox(label="输入文字")
    txt_message.submit(gen_img, [txt_message, imgs], [txt_message, gallery,imgs])

demo.queue(concurrency_count=1, max_size=20).launch(share=False, debug=True, server_port=7866,
                                                    server_name="0.0.0.0")

