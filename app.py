import streamlit as st
import re
from main import YouTubeSummarizer, SummaryContext

def is_valid_youtube_url(url):
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})'
    return bool(re.match(youtube_regex, url))

def main():
    st.title("YouTube Video Summarizer")
    st.write("Get AI-powered summaries tailored to your needs!")

    # Initialize the summarizer
    summarizer = YouTubeSummarizer(ollama_host='http://localhost:11434')

    # Fetch available models
    try:
        models = summarizer.get_available_models()
        summary_styles = summarizer.get_summary_styles()
    except Exception as e:
        st.error(f"Error connecting to Ollama server: {str(e)}")
        return

    # User inputs
    url = st.text_input("YouTube Video URL")
    
    # Model and basic style selection
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select AI Model", models)
    with col2:
        selected_style = st.selectbox(
            "Summary Style",
            list(summary_styles.keys()),
            help="Hover over options to see descriptions"
        )
        st.caption(summary_styles[selected_style])

    # Advanced options in an expander
    with st.expander("Advanced Options"):
        st.subheader("Customize Your Summary")
        
        # Purpose selection
        purpose = st.selectbox(
            "Summary Purpose",
            ["General Understanding", "Research", "Study/Exam Prep", "Teaching", "Business/Professional"]
        )
        
        # Audience selection
        audience = st.selectbox(
            "Target Audience",
            ["General", "Children", "Students", "Professionals", "Experts"]
        )
        
        # Formality slider
        formality = st.slider(
            "Formality Level",
            1, 5, 3,
            help="1: Very Casual, 5: Highly Formal"
        )
        
        # Detail level slider
        detail_level = st.slider(
            "Detail Level",
            1, 5, 3,
            help="1: High-level overview, 5: In-depth analysis"
        )

    if st.button("Generate Summary"):
        if not url:
            st.warning("Please enter a YouTube URL")
            return
        
        if not is_valid_youtube_url(url):
            st.error("Invalid YouTube URL")
            return

        try:
            # Create progress container
            progress_container = st.empty()
            
            # Extract transcript
            progress_container.info("Extracting transcript...")
            video_id = summarizer.get_video_id(url)
            transcript = summarizer.fetch_transcript(video_id)

            # Create context object
            context = SummaryContext(
                purpose=purpose,
                audience=audience,
                formality=formality,
                detail_level=detail_level
            )

            # Generate summary
            progress_container.info("Analyzing content and generating summary...")
            chunks = summarizer.split_transcript(transcript)
            summary = summarizer.summarize(
                chunks,
                selected_model,
                selected_style,
                context=context
            )
            
            # Clear progress and show results
            progress_container.empty()
            st.success("Summary generated!")
            
            # Display summary
            st.markdown("### Summary")
            st.markdown(summary)

            # Show transcript in expander
            with st.expander("View Full Transcript"):
                st.text(transcript)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()