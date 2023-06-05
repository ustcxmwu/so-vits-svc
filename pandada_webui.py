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
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE
from collections import deque

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging
import zmq
from multiprocessing import Process, Queue


class LogCollector(Process):
    """
    Centralized logger processor, a process used to collected log message sent by other processes.
    """

    def __init__(self, port, queue):
        super().__init__()
        self.port = port
        self.queue = queue

    def run(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:{}".format(self.port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.logger = logging.getLogger("LogCollector")
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)
        while True:
            topic, msg = self.socket.recv_multipart()
            self.queue.put((topic, msg))


log_queue = Queue()


def train_monitor():
    logs = deque(maxlen=100)
    while True:
        topic, msg = log_queue.get()
        msg = msg.decode().strip()
        print(msg)
        logs.put(msg)
        yield "\n".join(list(logs)[-10:])


def resample_dateset(dates):
    result = ["开始数据采样..."]
    yield "\n".join(result)
    p = Popen("python resample.py", shell=True, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    while True:
        out = p.stdout.readline()
        if out == '' and p.poll() is not None:
            break
        if out != "":
            result.append(out)
    result.append("数据采样完成.")
    yield "\n".join(result)


def split_dataset(speech_encoder):
    result = ["开始 Speech Encode..."]
    yield "\n".join(result)
    p = Popen(f"python preprocess_flist_config.py --speech_encoder {speech_encoder}", shell=True)
    p.wait()
    result.append("Speech Encode 结束.")
    yield "\n".join(result)


def f0_predict(predictor):
    result = ["开始 f0 Predict..."]
    yield "\n".join(result)
    p = Popen(f"python preprocess_hubert_f0.py --f0_predictor {predictor}", shell=True)
    p.wait()
    result.append("f0 Predict 结束.")
    yield "\n".join(result)


def train_diffusion_model():
    result = ["开始训练 Diffusion Model..."]
    yield "\n".join(result)
    p = Popen("python train_diff.py -c configs/diffusion.yaml", shell=True)
    p.wait()
    result.append("训练 Diffusion Model 结束.")
    yield "\n".join(result)


def train_sovits_model(total_epoch, batch_size):
    with open(Path("./configs/config.json"), mode='r') as f:
        config = json.load(f)
    config["train"]["epochs"] = int(total_epoch)
    config["train"]["batch_size"] = int(batch_size)
    # time_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    config_file = "config.json"
    with open(Path(f"./configs/{config_file}"), mode='w') as f:
        json.dump(config, f)
    p = Popen(f"python train.py -c configs/{config_file} -m 44k", shell=True)
    p.wait()


def covert_audio(model, audio_file, speaker, enhance, auto_f0, f0_predictor, vc_transform):
    raw_dir = Path("./raw")
    # audio_file = Path(audio_file)
    filename = Path(audio_file.name).name
    print(filename)
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
    if audio_file is not None:
        shutil.copy(audio_file.name, raw_dir / filename)
    cmd = f'python inference_main.py -m "logs/44k/{model}.pth" -c "configs/config.json" -n "{filename}" -t {int(vc_transform)} -s {speaker} --wav_format wav --f0_predictor {f0_predictor}'
    if auto_f0:
        cmd = cmd + " --auto_predict_f0"
    if enhance:
        cmd = cmd + " --enhance"

    result = ["开始音色迁移..."]
    yield None, "\n".join(result)
    p = Popen(cmd, shell=True)
    p.wait()
    result.append("结束音色迁移, 请查看转换后文件.")
    yield str(Path("./results") / f"{filename}_0key_{speaker}_sovits.wav"), "\n".join(result)


def upload_dataset(upload, dataset_name):
    if dataset_name == "":
        return
    dataset = Path("./dataset_raw") / dataset_name
    if not dataset.exists():
        dataset.mkdir(parents=True)
    for audio in upload:
        shutil.copy(audio.name, dataset / Path(audio.name).name)
    return get_dataset_info()


def get_dataset_info():
    dataset_info = []
    for e in Path("./dataset_raw").glob("*"):
        if e.is_dir():
            dataset_info.append((e.name, len([f for f in e.glob("*.wav")])))
    return "\n".join([f"Speaker: {d[0]}, 语料数量: {d[1]}" for d in dataset_info])


def get_models():
    models = [pt.stem for pt in Path("./logs/44k").glob('*.pth') if pt.name.startswith("G_")]
    return {"choices": models, "__type__": "update"}


with gr.Blocks() as app:
    with gr.Row(variant='panel').style(equal_height=True):
        with gr.Column(scale=1, min_width=80):
            gr.HTML("<html><img src='file/pandada.png', width=80, height=80 /><br></html>")
        with gr.Column(scale=30):
            gr.Markdown(
                """
                # 小白 Music Box (SoVITS)
                ##### Pandada Game 语音生成试验盒.
                """)
    with gr.Tabs():
        with gr.TabItem("训练"):
            with gr.Group():
                gr.Markdown(value="Step 1. Resample to 44100Hz and mono.")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            resample_btn = gr.Button("Resample")
                            resample_clear_btn = gr.Button("Clear")
                    with gr.Column():
                        resample_out = gr.Textbox(label="Resample 结果", lines=3, max_lines=3)
                    resample_btn.click(resample_dateset, None, resample_out)
                    resample_clear_btn.click(lambda: None, None, resample_out)
            with gr.Group():
                gr.Markdown(
                    value="Step 2. Split the dataset into training and validation sets, and generate configuration files.")
                with gr.Row():
                    with gr.Column():
                        speech_encoder_radio = gr.Radio(
                            label="选择 Speech Encoder",
                            choices=["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg"],
                            value="vec768l12",
                            interactive=True,
                        )
                        with gr.Row():
                            speech_encoder_btn = gr.Button("运行")
                    with gr.Column():
                        speech_encoder_out = gr.Textbox(label="运行结果", lines=3, max_lines=3)
                speech_encoder_btn.click(split_dataset, speech_encoder_radio, speech_encoder_out)
            with gr.Group():
                gr.Markdown(value="Step 3. Generate hubert and f0")
                with gr.Row():
                    with gr.Column():
                        f0_predictor_radio = gr.Radio(
                            label="选择 f0_predictor",
                            choices=["crepe", "dio", "pm", "harvest"],
                            value="dio",
                            interactive=True,
                        )
                        with gr.Row():
                            f0_predictor_btn = gr.Button("运行")
                    with gr.Column():
                        f0_predictor_out = gr.Textbox(label="运行结果", lines=3, max_lines=3)
                f0_predictor_btn.click(f0_predict, f0_predictor_radio, f0_predictor_out)
            # with gr.Group(visible=False):
            #     gr.Markdown(value="Step 4. (可选) 训练 Diffusion Model")
            #     with gr.Row():
            #         with gr.Column():
            #             with gr.Row():
            #                 train_diffusion_btn = gr.Button("训练 Diffusion Model")
            #                 train_diffusion_clear_btn = gr.Button("清除")
            #         with gr.Column():
            #             train_diffusion_out = gr.Textbox(label="Diffusion Model 模型训练结果", lines=5, max_lines=5)
            #         train_diffusion_btn.click(train_diffusion_model, None, train_diffusion_out)
            #         train_diffusion_clear_btn.click(lambda: None, None, train_diffusion_out)
            with gr.Group():
                gr.Markdown(value="Step 4. 训练 Sovits Model")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            total_epoch = gr.Slider(
                                minimum=0,
                                maximum=50000,
                                step=100,
                                label="总训练轮数total_epoch",
                                value=200,
                                interactive=True,
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=40,
                                step=1,
                                label="batch_size",
                                value=16,
                                interactive=True,
                            )
                        with gr.Row():
                            train_sovits_btn = gr.Button("训练 Sovits Model")
                            train_monitor_btn = gr.Button("查看训练日志")
                    with gr.Column(scale=2):
                        train_sovits_out = gr.Textbox(label="模型训练结果", lines=10, max_lines=10)
                    train_sovits_btn.click(train_sovits_model, [total_epoch, batch_size])
                    train_monitor_btn.click(train_monitor, None, train_sovits_out)

        with gr.TabItem("模型推理"):
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        # input_audio_file = gr.Audio(label="添加待转换音频", type="filepath")
                        input_audio_file = gr.File(label="添加待转换音频")
                        refresh_model_btn = gr.Button("刷新模型")
                        with gr.Row():
                            # experiments = gr.Dropdown(label="实验名称", choices=[""])
                            svc_model = gr.Dropdown(label="SVC 模型", choices=[""])
                            speakers = [f.name for f in Path("./dataset/44k").iterdir() if f.is_dir()]
                            speaker_name = gr.Dropdown(speakers, label="转换目标Speaker(文件夹 dataset_raw 中的子文件夹)")
                            refresh_model_btn.click(get_models, None, svc_model)
                        enhance = gr.Checkbox(
                            label="是否使用NSF_HIFIGAN增强,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭",
                            value=False)
                        auto_f0 = gr.Checkbox(
                            label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）",
                            value=False)
                        f0_predictor = gr.Dropdown(
                            label="选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)",
                            choices=["pm", "dio", "harvest", "crepe"], value="pm")
                        vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                        with gr.Row():
                            covert_btn = gr.Button("转换", variant="primary")
                            covert_clear_btn = gr.Button("清除", variant="primary")
                    with gr.Column():
                        covert_output_file = gr.Audio(label="输出音频(右下角三个点,点了可以下载)")
                        covert_log = gr.Textbox(label="音色迁移结果", lines=5, max_lines=5)
                    covert_btn.click(covert_audio, [svc_model, input_audio_file, speaker_name, enhance, auto_f0, f0_predictor, vc_transform],
                                     [covert_output_file, covert_log])
                    covert_clear_btn.click(lambda: None, None, [covert_output_file, covert_log])
        with gr.TabItem("训练集管理"):
            d_info = get_dataset_info()
            datasets = gr.Textbox(label="当前数据集", value=d_info, lines=5)
            dataset_name = gr.Textbox(label="输入训练数据集名称", value="speaker_name")
            upload_button = gr.UploadButton("上传训练数据", file_types=["audio"], file_count="multiple")
            upload_button.upload(upload_dataset, [upload_button, dataset_name], datasets, show_progress=True)

if __name__ == '__main__':
    collector = LogCollector(9999, log_queue)
    collector.start()

    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        server_port=7866,
        quiet=True,
        # root_path="./"
    )
    collector.join()
