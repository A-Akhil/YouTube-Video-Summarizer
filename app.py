# app.py
import streamlit as st
import re
from main import YouTubeSummarizer

def is_valid_youtube_url(url):
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})'
    return bool(re.match(youtube_regex, url))

def main():
    st.title("YouTube Video Summarizer")
    st.write("Enter a YouTube video URL and get an AI-powered summary!")

    # Initialize the summarizer
    summarizer = YouTubeSummarizer(ollama_host='http://localhost:11434')

    # Fetch available models
    try:
        models = summarizer.get_available_models()
    except Exception as e:
        st.error(f"Error connecting to Ollama server: {str(e)}")
        return

    # User inputs
    url = st.text_input("YouTube Video URL")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select AI Model", models)
    with col2:
        summary_style = st.selectbox(
            "Summary Style",
            ["Detailed Summary", "Concise Points", "Key Takeaways"]
        )

    if st.button("Generate Summary"):
        if not url:
            st.warning("Please enter a YouTube URL")
            return
        
        if not is_valid_youtube_url(url):
            st.error("Invalid YouTube URL")
            return

        try:
            with st.spinner("Extracting transcript..."):
                video_id = summarizer.get_video_id(url)
                transcript = summarizer.fetch_transcript(video_id)

            with st.spinner("Generating summary..."):
                chunks = summarizer.split_transcript(transcript)
                summary = summarizer.summarize(chunks, selected_model, summary_style)
                
                st.success("Summary generated!")
                st.markdown("### Summary")
                st.write(summary)

                with st.expander("View Full Transcript"):
                    st.text(transcript)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
