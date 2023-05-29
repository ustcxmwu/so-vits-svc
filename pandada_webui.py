#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   pandada_webui.py
@Time    :   2023-05-29 14:15
@Author  :   Xiaomin Wu <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2023, Xiaomin Wu
@Desc    :   Pandada Studio 的 svc 训练推理页面
"""
import gradio as gr


with gr.Blocks() as app:
    with gr.Row(variant='panel').style(equal_height=True):
        with gr.Column(scale=1, min_width=80):
            gr.HTML("<html><img src='file/pandada.png', width=80, height=80 /><br></html>")
        with gr.Column(scale=30):
            gr.Markdown(
                """
                # 小白 Music Box (SVC)
                ##### Pandada Game 语音生成试验盒.
                """)
    with gr.Tabs():
        with gr.TabItem("训练"):
            pass
        with gr.TabItem("模型推理"):
            pass
        with gr.TabItem("训练集管理"):
            pass
        # with gr.TabItem("常见问题解答"):
        #     pass

    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        server_port=7866,
        quiet=True,
    )
