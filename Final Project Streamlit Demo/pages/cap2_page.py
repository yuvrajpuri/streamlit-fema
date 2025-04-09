import streamlit as st
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
from streamlit_extras.mention import mention
from streamlit_extras.buy_me_a_coffee import button
import os

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "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"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-pt-224", trust_remote_code=True, token = HF_TOKEN)
    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-pt-224", device_map="auto", token = HF_TOKEN)
    return processor, model

processor, model = load_model()

@st.cache_data(show_spinner=False)
def generate_caption_from_bytes(img_bytes: bytes) -> str:
    image = Image.open(img_bytes).convert("RGB")
    image = preprocess_image(image)
    prompt = "<image> answer en "
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)

    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption

# Streamlit page
st.title("🧠 Image Captioning with PaliGemma")
st.caption("Experience the power of Google's multimodal language model.")

with st.expander("About this app"):
    st.markdown(
        """
        This app uses [PaliGemma](https://huggingface.co/google/paligemma2-3b-pt-224), a state-of-the-art vision-language model from Google, to generate natural language captions for images.

        Upload a cropped image and watch PaliGemma describe it like a pro!
        """
    )
    mention(label="Built with 🤗 Transformers", url="https://huggingface.co/transformers")

uploaded_file = st.file_uploader("Upload a cropped image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption using PaliGemma..."):
        caption = generate_caption_from_bytes(uploaded_file)

    st.success("📝 Caption:")
    st.markdown(f"> {caption}")

# Optional: show a small footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and 🤗 Transformers")
