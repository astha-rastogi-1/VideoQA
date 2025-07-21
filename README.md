# Video Question Answering with CLIP + BLIP + LLM

This project combines OpenAI's CLIP, BLIP for captioning, and a large language model (FLAN-T5) to perform question answering on short video clips.

### 🔍 Example:
**Input Video:** A man opens a fridge and takes out a soda.  
**Question:** What does the man do after opening the fridge?  
**Answer:** He takes out a soda.

## 🚀 Features
- Video frame sampling
- Frame selection using CLIP similarity
- Caption generation using BLIP
- Visual-text QA using FLAN-T5 (Open Source LLM)
- Demo UI via Gradio

## 🧠 Technologies Used
- PyTorch
- Hugging Face Transformers (BLIP, FLAN-T5)
- OpenCV, PIL
- Gradio (for demo UI)

## 🗂 Project Structure
```
video_qa_clip_llm/
├── app.py              # Gradio demo UI
├── video_qa.py         # Main logic for answering questions
├── utils.py            # Frame extraction
├── requirements.txt
├── sample_data/        # Sample videos + questions
├── README.md
```

## 📦 Setup
```bash
conda create -n videoqa python=3.9
conda activate videoqa
pip install -r requirements.txt
```

## ▶️ Run Demo
```bash
python app.py
```
"""