"""
YouTube Video Summarizer using official transcript API
Uses youtube-transcript-api to fetch transcripts legally and ethically
"""

import os
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import re

load_dotenv()


class EmbeddingModel:
    """Handles different embedding models"""

    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_type == "huggingface":
            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "nomic":
            self.embedding_fn = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    """Handles different LLM models"""

    def __init__(self, model_type="openai", model_name="gpt-4o-mini"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        elif model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YouTubeTranscriptSummarizer:
    def __init__(
            self,
            llm_type="openai",
            llm_model_name="gpt-4o-mini",
            embedding_type="openai"
    ):
        """
        Initialize with different LLM and embedding options

        Args:
            llm_type: 'openai' or 'ollama'
            llm_model_name: specific model name
            embedding_type: 'openai', 'huggingface', or 'nomic'
        """
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)

    def get_model_info(self) -> Dict:
        """Return current model configuration"""
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats

        Args:
            url: YouTube URL

        Returns:
            Video ID or None if not found
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def fetch_transcript(self, url: str) -> Dict:
        """
        Fetch transcript using YouTube's official API

        Args:
            url: YouTube video URL

        Returns:
            Dictionary containing transcript text and metadata
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        try:
            print(f"Fetching transcript for video ID: {video_id}")

            # Try to get transcript in English first, then any available language
            ytt_api = YouTubeTranscriptApi()
            try:
                transcript_list = ytt_api.fetch(
                    video_id=video_id,
                    languages=['en']
                )
            except NoTranscriptFound:
                # If English not available, get any available transcript
                transcript_list = ytt_api.fetch(video_id)

            # Combine all transcript segments into full text
            full_transcript = " ".join([entry.text for entry in transcript_list])

            # Get video metadata
            try:
                from pytube import YouTube
                yt = YouTube(url)
                video_title = yt.title
                video_author = yt.author
                video_length = yt.length
            except:
                # Fallback if pytube fails
                video_title = f"Video {video_id}"
                video_author = "Unknown"
                video_length = None

            return {
                "transcript": full_transcript,
                "video_id": video_id,
                "title": video_title,
                "author": video_author,
                "length": video_length,
                "url": url
            }

        except TranscriptsDisabled:
            raise Exception("Transcripts are disabled for this video")
        except NoTranscriptFound:
            raise Exception("No transcript found for this video")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    def create_documents(self, text: str, metadata: Dict) -> List[Document]:
        """
        Split text into chunks and create Document objects

        Args:
            text: Full transcript text
            metadata: Video metadata

        Returns:
            List of Document objects
        """
        print("Creating documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)

        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": metadata.get("title", "Unknown"),
                    "video_id": metadata.get("video_id", ""),
                    "author": metadata.get("author", "Unknown")
                }
            )
            for chunk in texts
        ]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create vector store from documents

        Args:
            documents: List of Document objects

        Returns:
            Chroma vector store
        """
        print(f"Creating vector store using {self.embedding_model.model_type} embeddings...")

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(self, documents: List[Document], summary_type: str = "detailed") -> str:
        """
        Generate summary using LangChain's summarize chain

        Args:
            documents: List of Document objects
            summary_type: 'detailed' or 'concise'

        Returns:
            Summary text
        """
        print(f"Generating {summary_type} summary...")

        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of this YouTube video transcript section:
            "{text}"

            Focus on the main points and key information discussed.
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a comprehensive summary of this YouTube video based on these section summaries:
            "{text}"

            Include:
            - Main topics and key points discussed
            - Important details, examples, and explanations
            - Any conclusions, recommendations, or calls to action
            - The overall message or purpose of the video

            FINAL DETAILED SUMMARY:"""
        else:
            map_prompt_template = """Write a concise summary of this YouTube video transcript section:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a brief summary of this YouTube video based on these section summaries:
            "{text}"

            FINAL CONCISE SUMMARY:"""

        map_prompt = ChatPromptTemplate.from_template(map_prompt_template)
        combine_prompt = ChatPromptTemplate.from_template(combine_prompt_template)

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        result = summary_chain.invoke(documents)
        return result.get("output_text", str(result))

    def setup_qa_chain(self, vector_store: Chroma):
        """
        Set up question-answering chain

        Args:
            vector_store: Chroma vector store

        Returns:
            ConversationalRetrievalChain
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True,
        )

    def process_video(self, url: str, summary_type: str = "detailed") -> Dict:
        """
        Process YouTube video and return summary and QA chain

        Args:
            url: YouTube video URL
            summary_type: 'detailed' or 'concise'

        Returns:
            Dictionary containing summary, QA chain, and metadata
        """
        try:
            # Fetch transcript
            transcript_data = self.fetch_transcript(url)

            # Create documents
            documents = self.create_documents(
                transcript_data["transcript"],
                transcript_data
            )

            # Generate summary
            summary = self.generate_summary(documents, summary_type)

            # Create vector store
            vector_store = self.create_vector_store(documents)

            # Setup QA chain
            qa_chain = self.setup_qa_chain(vector_store)

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": transcript_data["title"],
                "author": transcript_data["author"],
                "video_id": transcript_data["video_id"],
                "url": transcript_data["url"],
                "full_transcript": transcript_data["transcript"],
                "model_info": self.get_model_info()
            }

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {"error": str(e)}
