#!/usr/bin/env python
# encoding: utf-8
import gradio as gr
from PIL import Image

from VisCPM import VisCPMChat

# 修改你的模型地址
model_path = '/path/to/checkpoint'
viscpm_chat = VisCPMChat(model_path, image_safety_checker=False)
print("load  model  success !")


def upload_img(image,_chatbot,_app_session):
    image = Image.fromarray(image)
    _app_session['sts']=None
    _app_session['ctx']=''
    _app_session['img']=image
    _chatbot.append(('图片解析成功，可以和我对话了', ''))
    return _chatbot,_app_session


def respond( _question, _chat_bot,_app_cfg):
    _answer, _context, sts = viscpm_chat.chat(_app_cfg['img'], _question, _app_cfg['ctx'],
                                            vision_hidden_states=_app_cfg['sts'])
    _chat_bot.append((_question, _answer))
    _app_cfg['ctx']=_context
    _app_cfg['sts']=sts
    print('context', _context)
    return '',_chat_bot,_app_cfg


with gr.Blocks() as demo:
    app_session = gr.State({'sts':None,'ctx':None,'img':None})
    bt_pic = gr.Image(label="先上传一张图片")
    chat_bot = gr.Chatbot(label="聊天对话")
    txt_message = gr.Textbox(label="输入文字")

    txt_message.submit(respond, [ txt_message, chat_bot,app_session], [txt_message,chat_bot,app_session])
    bt_pic.upload(lambda: None, None, chat_bot, queue=False).then(upload_img, inputs=[bt_pic,chat_bot,app_session], outputs=[chat_bot,app_session])


demo.queue(concurrency_count=1, max_size=20).launch(share=False, debug=True, server_port=7866,
                                                    server_name="0.0.0.0")

