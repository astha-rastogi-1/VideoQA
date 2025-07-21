import torch
import clip
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize([""])

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)

    ## Check if video opens
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = [int(i * total_frames / num_frames) for i in range(num_frames)]    ## getting ids of num_frames frames evenly split out
    frames = []
    for i in range(total_frames):
        success, image = cap.read()
        if i in frame_idxs and success:
            frames.append(image)
    cap.release()
    return frames

def get_image_embedding(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input).squeeze(0)

def get_video_embedding(frames):
    inputs = torch.stack([
        preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames
    ]).to(device)
    with torch.no_grad():
        return model.encode_image(inputs).mean(dim=0).cpu()

def get_text_embedding(question):
    text = clip.tokenize([question]).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text).squeeze(0)
    return text_embeddings

def rank_frames_clip(frames, question, top_k=4):
    question_embedding = get_text_embedding(question)
    scored_frames = []
    for frame in frames:
        image_embedding = get_image_embedding(frame)
        similarity = torch.cosine_similarity(image_embedding, question_embedding, dim=0)
        scored_frames.append((similarity.item(), frame))
    
    scored_frames.sort(reverse=True, key=lambda x: x[0])
    return [f for _, f in scored_frames[:top_k]]

def generate_captions(frames):
    captions = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = caption_processor(images=image, return_tensors="pt").to(device)
        out = caption_model.generate(**inputs, max_new_tokens=20)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    print('CAPTIONS: \n', captions)
    return " ".join(captions)

def answer_question(video_path, question):
    frames = extract_frames(video_path)
    relevant_frames = rank_frames_clip(frames, question, top_k=4)
    context = generate_captions(relevant_frames)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    result = qa_pipeline(prompt, max_length=30)
    return result[0]['generated_text']

