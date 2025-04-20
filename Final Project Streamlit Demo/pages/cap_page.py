import streamlit as st

st.title("Captioning Tool (Coming Soon)")

st.markdown("This page will allow you to generate captions for disaster images using a vision-language model.")

# Placeholder for video walkthrough
st.header("🎥 How to Use This Tool (Video)")
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Replace with your real URL

# Placeholder for transcript in an expander
with st.expander("📄 Video Transcript"):
    st.markdown("""
    This will eventually include a full walkthrough of how to use the captioning tool...
    
    * Step 1: Upload an image
    * Step 2: Generate captions
    * Step 3: Export or copy the output
    """)

# Placeholder for external model link
st.header("📦 Model Access")
st.markdown("You can access the trained captioning model from [this link](https://huggingface.co/your-model) (coming soon).")
