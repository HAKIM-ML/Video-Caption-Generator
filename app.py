import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from decord import VideoReader, cpu
import numpy as np
import torch

# Load model and processor

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("model")
    return model
@st.cache_resource
def load_processor():
    processor = AutoProcessor.from_pretrained("input")
    return processor

model = load_model()
processor = load_processor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def sample_frames(file_path, num_frames):
    np.random.seed(45)

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    frames = videoreader.get_batch(indices).asnumpy()

    return list(frames)


st.title("Video Caption Generator")

# Sidebar with project details
st.sidebar.title("Project Details")
st.sidebar.write("Developed by: Md. Azizul Hakim")
st.sidebar.write("Institution: Bangladesh Sweden Polytechnic Institute")
st.sidebar.write("Department: CST")
st.sidebar.write("Semester: 5th")
st.sidebar.write("Project repository: [GitHub](https://github.com/HAKIM-ML)")
st.sidebar.write("---")
st.sidebar.write("### About the Project")
st.sidebar.write("This project is a video caption generator that uses a deep learning model to generate captions for uploaded videos. The model is trained on a dataset of videos and their corresponding captions, allowing it to learn the relationships between visual features and textual descriptions.")
st.sidebar.write("The user interface is built using the Streamlit framework, which provides an easy-to-use and interactive way to deploy the model. Users can upload a video, and the application will generate a caption for the video using the pre-trained model.")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display video
    st.video("temp_video.mp4")

    if st.button("Generate Caption"):
        with st.spinner('Generating caption...'):
            frames = sample_frames("temp_video.mp4", num_frames=6)
            inputs = processor(images=frames, return_tensors="pt").to(device)
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            st.write("Generated Caption:", caption)
