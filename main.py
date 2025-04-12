# -*- coding: utf-8 -*-
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
        # Environment variables to configure your OpenAI usage.
        self.llm_model = os.environ.get("MODEL")
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )

        # Check if CUDA is available; otherwise fallback to CPU.
        self.is_cuda = torch.cuda.is_available()

        # Load Whisper model once on CPU to save GPU memory
        # (pick any model size you want; "tiny", "base", "small", "medium", "large", etc.)
        self.whisper_model = whisper.load_model("turbo", device="cpu")

        # A lock to ensure only one thread at a time toggles model device
        # and calls transcribe
        self.whisper_processor_lock = threading.Lock()

    @staticmethod
    def download_youtube_sub_or_audio(video_url, output_path="outputs"):
        """Download subtitles if available, otherwise download audio."""
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

            # If subtitles exist, download them
            options = {}
            if has_subs:
                all_subs = {**subtitles}
                first_lang = next(iter(all_subs.keys()), None)
                if first_lang:
                    options.update({
                        'writesubtitles': True,  # download subtitles
                        'subtitleslangs': [first_lang],  # pick the first available language
                        'writeautomaticsub': True,  # include auto-generated subs
                        'outtmpl': f'{output_path}/subtitles_{info.get("id")}.%(ext)s',
                        'skip_download': True
                    })
                    with yt_dlp.YoutubeDL(options) as ydl:
                        video_info["subtitles_path"] = f'{output_path}/subtitles_{info.get("id")}.{first_lang}.vtt'
                        ydl.download([video_url])
                    return video_info

            # Otherwise, download audio
            options.update({
                'format': 'bestaudio/best',  # best audio quality
                'outtmpl': f'{output_path}/audio_{info.get("id")}.%(ext)s'
            })
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([video_url])

            # Find the audio file we just downloaded
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
        1) If we have an audio file, run Whisper transcribe.
        2) If we have a subtitles file, just read it directly.
        """
        # Case 1: We have an audio path
        if video_info["audio_path"]:
            try:
                with self.whisper_processor_lock:
                    # Move model to CUDA if available
                    if self.is_cuda:
                        self.whisper_model.to("cuda")

                    # Transcribe the audio
                    result = self.whisper_model.transcribe(video_info["audio_path"], verbose=False)

                    # Move model back to CPU to free GPU memory
                    if self.is_cuda:
                        self.whisper_model.to("cpu")
                        # Optionally clear PyTorch GPU cache
                        torch.cuda.empty_cache()

                return {"status": "success", "text": result["text"]}
            except Exception as e:
                return {"status": "error", "text": f"whisper error: {e}"}

        # Case 2: We have a subtitles file
        elif video_info["subtitles_path"]:
            try:
                with open(video_info["subtitles_path"], "r", encoding="utf-8") as f:
                    sub_info = f.read().split("\n")
                # Remove lines that contain '-->'
                sub_info = [line for line in sub_info if '-->' not in line]
                return {"status": "success", "text": "\n".join(sub_info)}
            except Exception as e:
                return {"status": "error", "text": f"读取字幕文件出错: {e}"}

        # Case 3: Error
        elif video_info["error_info"]:
            return {"status": "error", "text": video_info["error_info"]}

        # Case 4: Unknown
        else:
            return {"status": "error", "text": "未知错误"}

    def summary_text_url(self, title, text):
        """
        Summarize transcribed text from a URL-based audio/video.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
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
        Summarize transcribed text from an uploaded audio file.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "总结录音，简体中文回答"},
                    {"role": "user", "content": text},
                ],
                stream=False
            )
            return "success", response.choices[0].message.content
        except Exception as e:
            return "error", f"OpenAI 接口错误: {e}"


class UI:
    def __init__(self):
        self.audio_summarizer = AudioSummarizer()
        self.tasks = {}
        self.current_task_number = 0

    @staticmethod
    def is_valid_url(url):
        # Just a simple check for known domains:
        valid_domains = ["youtube.com", "youtu.be", "bilibili.com", "b23.tv"]
        regex = re.compile(
            r'^(https?://)?(www\.)?(' + '|'.join(re.escape(domain) for domain in valid_domains) + r')/.+$'
        )
        return bool(regex.match(url))

    def update_table(self):
        table_data = []
        for task_number, task in self.tasks.items():
            table_data.append([task["title"]])
        # Reverse so newest is on top
        return table_data[::-1]

    def get_summary(self, evt: gr.SelectData):
        """
        When the user clicks a row in the DataFrame, we return the summary and the original text.
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

    def handle_url_task(self, youtube_url, task_number):
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

    def add_url_task(self, youtube_url):
        if not youtube_url or not self.is_valid_url(youtube_url):
            return self.update_table(), "不正确的url"

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": youtube_url,
            "url": youtube_url,
            "status": "未完成",
            "summary": "",
            "origin_text": "",
            "type": "",
        }

        # Start in a separate thread so it doesn't block the UI
        threading.Thread(
            target=self.handle_url_task,
            args=(youtube_url, self.current_task_number),
            daemon=True
        ).start()

        self.current_task_number += 1
        return self.update_table(), ""

    def add_file_task(self, file):
        if file is None:
            return self.update_table()

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_title = os.path.basename(file.name)
        self.tasks[self.current_task_number] = {
            "time": cur_time,
            "title": file_title,
            "url": file.name,
            "status": "未完成",
            "summary": "",
            "origin_text": "",
            "type": "",
        }

        # Start in a separate thread
        threading.Thread(
            target=self.handle_file_task,
            args=(file, self.current_task_number),
            daemon=True
        ).start()

        self.current_task_number += 1
        return self.update_table()

    def handle_file_task(self, file, task_number):
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

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("## YouTube/Bilibili/音频 自动总结")

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
                            url_add_btn = gr.Button("添加视频任务", size="lg", variant="primary")
                            file_add_btn = gr.Button("添加音频任务", size="lg", variant="primary")

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
                        max_lines=20,
                    )

                with gr.Column(scale=6):
                    gr.Markdown("### 摘要")
                    summary_md = gr.Markdown(
                        "请在左侧点击任务后在此处查看摘要。"
                    )

            # Button/link actions:
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
            tasks_table.select(
                fn=self.get_summary,
                inputs=[],
                outputs=[summary_md, origin_text]
            )

            # Initialize table
            demo.load(
                fn=self.update_table,
                inputs=[],
                outputs=tasks_table
            )

        demo.launch(server_name="0.0.0.0", show_api=False)


if __name__ == "__main__":
    UI().launch()
