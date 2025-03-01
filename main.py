# -*- coding: utf-8 -*-
import gc
import os
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
        self.llm_model = os.environ.get("MODEL")
        self.client = OpenAI(api_key=os.environ.get("API_KEY"),
                             base_url=os.environ.get("BASE_URL")
                             )

        self.is_cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.whisper_user_number = 0
        self.whisper_user_lock = threading.Lock()
        self.whisper_processor_lock = threading.Lock()

    @staticmethod
    def download_youtube_sub_or_audio(video_url, output_path="outputs"):
        # 第一步: 获取视频信息, 检查是否存在字幕
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

            # 第二步: 根据字幕情况设置下载选项
            options = {}

            if has_subs:
                all_subs = {**subtitles}
                first_lang = next(iter(all_subs.keys()), None)
                if first_lang:
                    options.update({
                        'writesubtitles': True,  # 启用字幕下载
                        'subtitleslangs': [first_lang],  # 选择第一个语言的字幕
                        'writeautomaticsub': True,  # 包括自动生成的字幕
                        'outtmpl': f'{output_path}/subtitles_{info.get("id")}.%(ext)s',  # 设置输出文件名
                        'skip_download': True  # 跳过视频下载
                    })
                    # 执行下载
                    with yt_dlp.YoutubeDL(options) as ydl:
                        video_info["subtitles_path"] = f'{output_path}/subtitles_{info.get("id")}.{first_lang}.vtt'
                        ydl.download([video_url])
                    return video_info

            # 第三步: 如果没有字幕, 下载音频
            options.update({
                'format': 'bestaudio/best',  # 下载最佳音频质量
                'outtmpl': f'{output_path}/audio_{info.get("id")}.%(ext)s'  # 设置输出文件名
            })
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([video_url])

            # 列出文件夹中的所有文件，文件名包含{output_path}/audio_{info.get("id")}
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.startswith(f"audio_{info.get('id')}"):
                        video_info["audio_path"] = os.path.join(root, file)
                        return video_info

            video_info["error_info"] = "音频字幕下载失败"
            return video_info
        except Exception as e:
            video_info["error_info"] = str(e)
            return video_info

    def extract_info_from_sub_or_audio(self, video_info):
        if video_info["audio_path"]:
            try:
                with self.whisper_user_lock:
                    if self.whisper_user_number == 0:
                        self.whisper_model = whisper.load_model("turbo", device=self.is_cuda)
                    self.whisper_user_number += 1

                with self.whisper_processor_lock:
                    result = self.whisper_model.transcribe(video_info["audio_path"], verbose=False)

                with self.whisper_user_lock:
                    self.whisper_user_number -= 1
                    if self.whisper_user_number == 0:
                        del self.whisper_model
                        if self.is_cuda == "cuda":
                            gc.collect()
                            torch.cuda.empty_cache()

                return {"status": "success", "text": result["text"]}
            except Exception as e:
                print(e)
                return {"status": "error", "text": "whisper error " + str(e)}
        elif video_info["subtitles_path"]:
            with open(video_info["subtitles_path"], "r", encoding="utf-8") as f:
                sub_info = f.read().split("\n")
                for line in sub_info:
                    if "-->" in line:
                        sub_info.remove(line)
                return {"status": "success", "text": "\n".join(sub_info)}
        elif video_info["error_info"]:
            return {"status": "error", "text": video_info["error_info"]}
        else:
            return {"status": "error", "text": "未知错误"}

    def summary_text_url(self, title, text):
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": f"本次录音的标题是{title}, 简要回答标题的问题，并且总结录音，简体中文回答"},
                    {"role": "user", "content": text},
                ],
                stream=False
            )
            return "success", response.choices[0].message.content
        except Exception as e:
            return "error", "OpenAI 接口错误: " + str(e)

    def summary_text_audio(self, text):
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": f"总结录音，简体中文回答"},
                    {"role": "user", "content": text},
                ],
                stream=False
            )
            return "success", response.choices[0].message.content
        except Exception as e:
            return "error", "OpenAI 接口错误: " + str(e)


class UI:
    def __init__(self):
        self.audio_summarizer = AudioSummarizer()
        self.tasks = {}
        self.current_task_number = 0

    @staticmethod
    def is_valid_url(url):
        valid_domains = ["youtube.com", "youtu.be", "bilibili.com", "b23.tv"]
        regex = re.compile(
            r'^(https?://)?(www\.)?(' + '|'.join(re.escape(domain) for domain in valid_domains) + r')/.+$'
        )
        return regex.match(url) is not None

    def update_table(self):
        table_data = []
        for task_number, task in self.tasks.items():
            table_data.append([task["title"]])

        return table_data[::-1]

    def get_summary(self, evt: gr.SelectData):
        # return f"You selected {evt.value} at {evt.index} from {evt.target}"
        title = evt.value
        for task in self.tasks.values():
            if task["title"] == title:
                if task["type"] == "url":
                    if task["summary"] == '':
                        return task["status"]
                    else:
                        return f'## [{task["title"]}]({task["url"]}) \n\n {task["summary"]}'
                else:
                    if task["summary"] == '':
                        return task["status"]
                    else:
                        return f'## {task["title"]} \n\n {task["summary"]}'

        return "未找到摘要"

    def handle_url_task(self, youtube_url, task_number):
        self.tasks[task_number]["type"] = "url"
        self.tasks[task_number]["status"] = "获取视频信息中..."
        video_info = self.audio_summarizer.download_youtube_sub_or_audio(youtube_url)
        self.tasks[task_number]["title"] = video_info["title"]
        if video_info["error_info"]:
            self.tasks[task_number]["summary"] = video_info["error_info"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在语音转文本"
        text_result = self.audio_summarizer.extract_info_from_sub_or_audio(video_info)

        if text_result["status"] == "error":
            self.tasks[task_number]["summary"] = text_result["text"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在总结"
        final_summary = self.audio_summarizer.summary_text_url(video_info["title"], text_result["text"])

        self.tasks[task_number]["summary"] = final_summary[1]
        self.tasks[task_number]["status"] = final_summary[0]

    def add_url_task(self, youtube_url):
        if youtube_url == "":
            return self.update_table(), "不正确的url"

        if not self.is_valid_url(youtube_url):
            return self.update_table(), "不正确的url"

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": youtube_url,  # 后台线程处理完成后更新为实际视频标题
            "url": youtube_url,
            "status": "未完成",  # 初始任务状态
            "summary": ""  # 后台处理完成后存储摘要
        }
        threading.Thread(target=self.handle_url_task, args=(youtube_url, self.current_task_number), daemon=True).start()
        self.current_task_number += 1
        return self.update_table(), ""

    def add_file_task(self, file):
        if file is None:
            return self.update_table()

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": file.name.split("/")[-1],  # 后台线程处理完成后更新为实际标题
            "url": file.name,
            "status": "未完成",  # 初始任务状态
            "summary": ""  # 后台处理完成后存储摘要
        }
        threading.Thread(target=self.handle_file_task, args=(file, self.current_task_number), daemon=True).start()
        self.current_task_number += 1
        return self.update_table()

    def handle_file_task(self, file, task_number):
        self.tasks[task_number]["type"] = "file"
        self.tasks[task_number]["status"] = "正在语音转文本"
        text_result = self.audio_summarizer.extract_info_from_sub_or_audio({"audio_path": file.name})

        if text_result["status"] == "error":
            self.tasks[task_number]["summary"] = text_result["text"]
            self.tasks[task_number]["status"] = "失败"
            return

        self.tasks[task_number]["status"] = "正在总结"
        final_summary = self.audio_summarizer.summary_text_audio(text_result["text"])

        self.tasks[task_number]["summary"] = final_summary[1]
        self.tasks[task_number]["status"] = final_summary[0]

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("## YouTube/Bilibili/音频 自动总结")
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column():
                        with gr.Row():
                            url_input = gr.Textbox(label="请输入视频链接", placeholder="https://www.youtube.com/...")
                            file_input = gr.File(label="上传音频文件", file_types=[".mp3", ".wav", ".m4a", ".mp4"],
                                                 height=88)
                        with gr.Row():
                            url_add_btn = gr.Button("添加视频任务", size="lg", variant="primary")
                            file_add_btn = gr.Button("添加音频任务", size="lg", variant="primary")

                    refresh_btn = gr.Button("刷新任务列表")
                    tasks_table = gr.DataFrame(
                        headers=["视(音)频标题"],
                        interactive=False,
                        label="任务概览 (点击标题查看摘要)",
                        max_height=300,
                    )

                with gr.Column(scale=6):
                    gr.Markdown("### 摘要")
                    summary_md = gr.Markdown("请在左侧点击视频标题后在此处显示摘要。")

            url_add_btn.click(fn=self.add_url_task, inputs=url_input, outputs=[tasks_table, url_input])
            file_add_btn.click(fn=self.add_file_task, inputs=file_input, outputs=tasks_table)
            refresh_btn.click(fn=self.update_table, inputs=[], outputs=tasks_table)
            tasks_table.select(fn=self.get_summary, inputs=[], outputs=summary_md)

            demo.load(fn=self.update_table, inputs=[], outputs=tasks_table)

        demo.launch(server_name="0.0.0.0", show_api=False)


if __name__ == "__main__":
    UI().launch()
