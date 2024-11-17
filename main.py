from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import re

class YouTubeSummarizer:
    def __init__(self, ollama_host):
        """Initialize the summarizer with Ollama host URL."""
        self.ollama_host = ollama_host
        self.client = Client(host=ollama_host)
    
    def get_video_id(self, url):
        """Extract video ID from YouTube URL."""
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def fetch_transcript(self, video_id):
        """Fetch and format the video transcript."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            return formatter.format_transcript(transcript)
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    def split_transcript(self, transcript, chunk_size=2048):
        """Split transcript into manageable chunks."""
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        return text_splitter.create_documents([transcript])

    def get_available_models(self):
        """Get list of available Ollama models."""
        try:
            return [model["name"] for model in self.client.list()["models"]]
        except Exception as e:
            raise Exception(f"Error fetching models: {str(e)}")

    def get_summary_prompt(self, style):
        """Get appropriate prompt template based on summary style."""
        prompts = {
            "Detailed Summary": """Create a detailed summary of this video segment. Focus on the main points 
                               and supporting details. Begin your response with 'This segment covers' or 
                               'This portion discusses': {chunk}""",
            
            "Concise Points": """Create a concise summary of the key points from this video segment. 
                             Focus only on the most important information. Format as 2-3 clear sentences: {chunk}""",
            
            "Key Takeaways": """Extract the 1-2 most important takeaways from this video segment. 
                            Format as brief, clear bullet points: {chunk}"""
        }
        return prompts.get(style, prompts["Detailed Summary"])

    def combine_summaries(self, summaries, style):
        """Combine chunk summaries based on the chosen style."""
        if style == "Key Takeaways":
            return "\n".join(f"• {summary.strip()}" for summary in summaries)
        return " ".join(summaries)

    def summarize(self, chunks, model_name, style="Detailed Summary"):
        """Generate summary using the selected model and style."""
        try:
            llm = ChatOllama(
                model=model_name,
                temperature=0.5,
                base_url=self.ollama_host
            )
            prompt = ChatPromptTemplate.from_template(self.get_summary_prompt(style))
            chain = prompt | llm | StrOutputParser()

            # Process chunks and combine summaries
            summaries = [chain.invoke({"chunk": chunk.page_content}) for chunk in chunks]
            
            # For longer content, summarize the summaries
            if len(summaries) > 3 and style != "Key Takeaways":
                combined = self.combine_summaries(summaries, style)
                final_prompt = ChatPromptTemplate.from_template(
                    "Synthesize these summary points into a coherent summary: {text}"
                )
                final_chain = final_prompt | llm | StrOutputParser()
                return final_chain.invoke({"text": combined})
            
            return self.combine_summaries(summaries, style)

        except Exception as e:
            raise Exception(f"Error during summarization: {str(e)}")