"""
Gradio UI
"""

import gradio as gr
from videoqa import answer_question

def qa_interface(video, question):
    # print(video)
    # video_path = "temp_video.mp4"
    # print(type(video))
    # video.save(video_path)
    return answer_question(video, question)

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=qa_interface,
    inputs=["video", "text"],
    outputs=["text"],
    title="Video Question Answering",
    description="Upload a short video and ask a question about its content."
)

demo.launch()