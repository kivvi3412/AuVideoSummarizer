# -*- coding: utf-8 -*-
import os
import queue
import re
import threading
from datetime import datetime

import gradio as gr
import torch
import whisper
import yt_dlp
from openai import OpenAI


class AudioSummarizer:
    def __init__(self):
        """
        初始化时不加载Whisper模型
        """
        self.model_name = None
        self.whisper_model = None
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )

        # 是否能使用CUDA
        self.is_cuda = torch.cuda.is_available()

        # 用于在推理时串行处理
        self.whisper_processor_lock = threading.Lock()

    def get_model_status(self):
        """
        获取当前加载的模型状态
        """
        if self.model_name is None:
            return "NONE"
        return self.model_name

    def load_whisper_model(self, model_name: str):
        """
        加载指定名字的Whisper模型到GPU上。
        如果已有模型则先卸载再加载。
        """
        if model_name == "NONE":
            self.unload_whisper_model()
            return "NONE"

        if self.whisper_model is not None:
            self.unload_whisper_model()

        self.model_name = model_name
        # 保持模型常驻GPU
        device = "cuda" if self.is_cuda else "cpu"
        self.whisper_model = whisper.load_model(model_name, device=device)
        return model_name

    def unload_whisper_model(self):
        """
        卸载当前加载的Whisper模型。
        """
        self.whisper_model = None
        self.model_name = None
        # 如果使用了GPU，可以主动清理显存
        if self.is_cuda:
            torch.cuda.empty_cache()

    @staticmethod
    def download_youtube_sub_or_audio(video_url, output_path="outputs"):
        """下载字幕(若有)，否则下载音频。"""
        video_info = {
            "id": None,
            "title": None,
            "webpage_url": None,
            "subtitles_path": None,
            "audio_path": None,
            "error_info": None
        }
        try:
            with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subtitles = info.get('subtitles', {})
                has_subs = len(subtitles) > 0 and "live_chat" not in subtitles

                video_info["id"] = info.get('id')
                video_info["title"] = info.get('title')
                video_info["webpage_url"] = info.get('webpage_url')

            # 如果有字幕则下载字幕
            options = {}
            if has_subs:
                all_subs = {**subtitles}
                first_lang = next(iter(all_subs.keys()), None)
                if first_lang:
                    options.update({
                        'writesubtitles': True,  # 下载字幕
                        'subtitleslangs': [first_lang],  # 只下载第一种可用语言
                        'writeautomaticsub': True,  # 包括自动生成的字幕
                        'outtmpl': f'{output_path}/subtitles_{info.get("id")}.%(ext)s',
                        'skip_download': True
                    })
                    with yt_dlp.YoutubeDL(options) as ydl:
                        video_info["subtitles_path"] = f'{output_path}/subtitles_{info.get("id")}.{first_lang}.vtt'
                        ydl.download([video_url])
                    return video_info

            # 否则下载音频
            options.update({
                'format': 'bestaudio/best',
                'outtmpl': f'{output_path}/audio_{info.get("id")}.%(ext)s'
            })
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([video_url])

            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.startswith(f"audio_{info.get('id')}"):
                        video_info["audio_path"] = os.path.join(root, file)
                        return video_info

            video_info["error_info"] = "音频或字幕下载失败"
            return video_info
        except Exception as e:
            video_info["error_info"] = str(e)
            return video_info

    def extract_info_from_sub_or_audio(self, video_info):
        """
        1) 有音频文件则调用Whisper识别
        2) 有字幕文件则直接读取
        3) 若报错则返回错误信息
        """
        # 没有加载模型时直接返回错误
        if self.whisper_model is None:
            return {"status": "error", "text": "Whisper模型尚未加载，请先加载模型"}

        # Case 1: 有音频
        if video_info["audio_path"]:
            try:
                with self.whisper_processor_lock:
                    # 直接在GPU或CPU上推理，不再来回切换
                    result = self.whisper_model.transcribe(video_info["audio_path"], verbose=False)
                return {"status": "success", "text": result["text"]}
            except Exception as e:
                return {"status": "error", "text": f"whisper error: {e}"}

        # Case 2: 有字幕文件
        elif video_info["subtitles_path"]:
            try:
                with open(video_info["subtitles_path"], "r", encoding="utf-8") as f:
                    sub_info = f.read().split("\n")
                # 去掉包含'-->'的行
                sub_info = [line for line in sub_info if '-->' not in line]
                return {"status": "success", "text": "\n".join(sub_info)}
            except Exception as e:
                return {"status": "error", "text": f"读取字幕文件出错: {e}"}

        # Case 3: 报错信息
        elif video_info["error_info"]:
            return {"status": "error", "text": video_info["error_info"]}

        # Case 4: 未知错误
        else:
            return {"status": "error", "text": "未知错误"}

    def summary_text_url(self, title, text):
        """
        利用OpenAI对URL的文字内容总结
        """
        if not self.llm_model_ready():
            return "error", "OpenAI模型尚未配置或出现错误"

        try:
            response = self.client.chat.completions.create(
                model=os.environ.get("MODEL"),  # 这里依旧使用环境变量MODEL
                messages=[
                    {
                        "role": "system",
                        "content": f"本次录音的标题是{title}, 简要回答标题的问题，并且总结录音，简体中文回答"
                    },
                    {
                        "role": "user",
                        "content": text
                    },
                ],
                stream=False
            )
            return "success", response.choices[0].message.content
        except Exception as e:
            return "error", f"OpenAI 接口错误: {e}"

    def summary_text_audio(self, text):
        """
        利用OpenAI对上传音频的文字内容总结
        """
        if not self.llm_model_ready():
            return "error", "OpenAI模型尚未配置或出现错误"

        try:
            response = self.client.chat.completions.create(
                model=os.environ.get("MODEL"),  # 这里依旧使用环境变量MODEL
                messages=[
                    {"role": "system", "content": "总结录音，简体中文回答"},
                    {"role": "user", "content": text},
                ],
                stream=False
            )
            return "success", response.choices[0].message.content
        except Exception as e:
            return "error", f"OpenAI 接口错误: {e}"

    def llm_model_ready(self):
        """
        判断是否已经在环境变量里正确配置了OpenAI MODEL, 以及api_key
        """
        if not os.environ.get("MODEL") or not os.environ.get("API_KEY"):
            return False
        return True


class UI:
    def __init__(self):
        self.audio_summarizer = AudioSummarizer()

        self.tasks = {}  # 记录任务信息
        self.current_task_number = 0

        # 准备一个队列，所有任务统一进入队列
        self.task_queue = queue.Queue()

        # 启动后台线程，从队列中逐一取出任务处理
        self.worker_thread = threading.Thread(target=self.process_tasks, daemon=True)
        self.worker_thread.start()
        self.current_model_name = "NONE"

    def process_tasks(self):
        """
        单一后台线程，从队列中逐个取出任务，并依次处理。
        """
        while True:
            task_id, task_data = self.task_queue.get()  # (task_number, { ... })
            task_type = task_data["type"]
            if task_type == "url":
                self.handle_url_task(task_id, task_data["url"])
            elif task_type == "file":
                self.handle_file_task(task_id, task_data["file"])
            self.task_queue.task_done()

    @staticmethod
    def is_valid_url(url):
        """
        简单验证URL是否来自youtube/bilibili
        """
        valid_domains = ["youtube.com", "youtu.be", "bilibili.com", "b23.tv"]
        regex = re.compile(
            r'^(https?://)?(www\.)?(' + '|'.join(re.escape(domain) for domain in valid_domains) + r')/.+$'
        )
        return bool(regex.match(url))

    def update_table(self):
        """
        根据self.tasks的内容更新数据表
        """
        table_data = []
        for task_number, task in self.tasks.items():
            table_data.append([task["title"]])
        # 最新任务置顶
        return table_data[::-1]

    def get_summary(self, evt: gr.SelectData):
        """
        当用户在DataFrame中点击某一行时，显示摘要和原始文本
        """
        title = evt.value
        for task in self.tasks.values():
            if task["title"] == title:
                if task["type"] == "url":
                    if task["summary"] == '':
                        return task["status"], task["origin_text"]
                    else:
                        return f'## [{task["title"]}]({task["url"]}) \n\n {task["summary"]}', task["origin_text"]
                else:  # file
                    if task["summary"] == '':
                        return task["status"], task["origin_text"]
                    else:
                        return f'## {task["title"]} \n\n {task["summary"]}', task["origin_text"]

        return "未找到摘要", ""

    def handle_url_task(self, task_number, youtube_url):
        """
        真正执行URL任务的逻辑(在后台线程里顺序执行)
        """
        self.tasks[task_number]["type"] = "url"
        self.tasks[task_number]["status"] = "获取视频信息中..."

        video_info = self.audio_summarizer.download_youtube_sub_or_audio(youtube_url)
        self.tasks[task_number]["title"] = video_info["title"] or youtube_url

        if video_info["error_info"]:
            self.tasks[task_number]["summary"] = video_info["error_info"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在语音转文本"
        text_result = self.audio_summarizer.extract_info_from_sub_or_audio(video_info)
        self.tasks[task_number]["origin_text"] = text_result["text"]

        if text_result["status"] == "error":
            self.tasks[task_number]["summary"] = text_result["text"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在总结"
        final_summary = self.audio_summarizer.summary_text_url(
            video_info["title"], text_result["text"]
        )

        self.tasks[task_number]["summary"] = final_summary[1]
        self.tasks[task_number]["status"] = final_summary[0]

    def handle_file_task(self, task_number, file):
        """
        真正执行文件任务的逻辑(在后台线程里顺序执行)
        """
        self.tasks[task_number]["type"] = "file"
        self.tasks[task_number]["status"] = "正在语音转文本"

        text_result = self.audio_summarizer.extract_info_from_sub_or_audio(
            {"audio_path": file.name}
        )
        self.tasks[task_number]["origin_text"] = text_result["text"]

        if text_result["status"] == "error":
            self.tasks[task_number]["summary"] = text_result["text"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在总结"
        final_summary = self.audio_summarizer.summary_text_audio(text_result["text"])

        self.tasks[task_number]["summary"] = final_summary[1]
        self.tasks[task_number]["status"] = final_summary[0]

    def add_url_task(self, youtube_url):
        """
        将URL任务放进队列。
        """
        if not youtube_url or not self.is_valid_url(youtube_url):
            return self.update_table(), "不正确的url"

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": youtube_url,
            "url": youtube_url,
            "status": "排队中",
            "summary": "",
            "origin_text": "",
            "type": "",
        }

        # 将任务加入队列
        self.task_queue.put((self.current_task_number, {"type": "url", "url": youtube_url}))
        self.current_task_number += 1
        return self.update_table(), ""

    def add_file_task(self, file):
        """
        将文件任务放进队列。
        """
        if file is None:
            return self.update_table()

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_title = os.path.basename(file.name)
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": file_title,
            "url": file.name,
            "status": "排队中",
            "summary": "",
            "origin_text": "",
            "type": "",
        }

        # 将任务加入队列
        self.task_queue.put((self.current_task_number, {"type": "file", "file": file}))
        self.current_task_number += 1
        return self.update_table()

    def load_model(self, chosen_model):
        """
        加载指定名称的Whisper模型到GPU
        """
        if chosen_model == self.current_model_name:
            return self.current_model_name
        else:
            try:
                self.current_model_name = chosen_model
                return self.audio_summarizer.load_whisper_model(chosen_model)
            except Exception as e:
                self.current_model_name = "NONE"
                return "NONE"

    def get_current_model(self):
        """
        获取当前加载的模型名称
        """
        return self.audio_summarizer.get_model_status()

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("## YouTube/Bilibili/音频 自动总结 (单任务队列)")

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column():
                        with gr.Row():
                            url_input = gr.Textbox(
                                label="请输入视频链接",
                                placeholder="https://www.youtube.com/..."
                            )
                            file_input = gr.File(
                                label="上传音频文件",
                                file_types=[".mp3", ".wav", ".m4a", ".mp4"],
                                height=88
                            )
                        with gr.Row():
                            url_add_btn = gr.Button("添加视频任务", variant="primary")
                            file_add_btn = gr.Button("添加音频任务", variant="primary")
                    with gr.Row():
                        model_choices = ["NONE", "tiny", "base", "small", "medium", "turbo"]
                        model_select = gr.Dropdown(
                            model_choices, value="NONE", label="Whisper模型选择 (当前模型状态)",
                        )
                        refresh_btn = gr.Button("刷新任务列表")
                    tasks_table = gr.DataFrame(
                        headers=["视(音)频标题"],
                        interactive=False,
                        label="任务概览 (点击标题查看摘要)",
                        max_height=300,
                    )
                    origin_text = gr.Textbox(
                        label="原始文本",
                        placeholder="请在左侧点击视频标题后查看原始文本。",
                        max_lines=20
                    )

                with gr.Column(scale=6):
                    gr.Markdown("### 摘要")
                    summary_md = gr.Markdown(
                        "请在左侧点击任务后在此处查看摘要。"
                    )

            # 模型选择事件处理 - 切换模型时自动卸载旧模型并加载新模型
            model_select.change(
                fn=self.load_model,
                inputs=model_select,
                outputs=model_select
            )

            # 任务相关按钮
            url_add_btn.click(
                fn=self.add_url_task,
                inputs=url_input,
                outputs=[tasks_table, url_input]
            )
            file_add_btn.click(
                fn=self.add_file_task,
                inputs=file_input,
                outputs=tasks_table
            )
            refresh_btn.click(
                fn=self.update_table,
                inputs=[],
                outputs=tasks_table
            )

            # 点选任务，查看摘要
            tasks_table.select(
                fn=self.get_summary,
                inputs=[],
                outputs=[summary_md, origin_text]
            )

            # 初始化表并获取当前模型状态
            demo.load(
                fn=lambda: (self.update_table(), self.get_current_model()),
                inputs=[],
                outputs=[tasks_table, model_select]
            )

            demo.launch(server_name="0.0.0.0", show_api=False)


if __name__ == "__main__":
    UI().launch()
