import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


# Setup LLMs
class GroqLLM:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def get_llm(self):
        try:
            os.environ["GROQ_API_KEY"]=self.groq_api_key
            llm = ChatGroq(api_key=self.groq_api_key, model="llama-3.1-8b-instant")
            return llm
        except Exception as e:
            raise ValueError("Error occurred with exception : {e}")


# Setup States
class Blog(BaseModel):
    title: str = Field(description="the title of the blog post")
    content: str = Field(description="The main content of the blog post")


class BlogState(TypedDict):
    topic: str
    blog: Blog
    current_language: str


# Setup Nodes
class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt = """
                   You are an expert blog content writer. Use Markdown formatting. Generate
                   a blog title for the {topic}. This title should be creative and SEO friendly

                   """

            sytem_message = prompt.format(topic=state["topic"])
            print(sytem_message)
            response = self.llm.invoke(sytem_message)
            print(response)
            return {"blog": {"title": response.content}}
        else:
            return {"blog": {"title": ""}}

    def content_generation(self, state: BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}"""
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}
        else:
            return {"blog": {"title": state['blog']['title'], "content": ""}}

    def translation(self, state: BlogState):
        """
        Translate the content to the specified language.
        """
        translation_prompt = """
        Translate the following content into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.

        ORIGINAL CONTENT:
        {blog_content}

        """
        print(state["current_language"])
        blog_content = state["blog"]["content"]
        messages = [
            HumanMessage(
                translation_prompt.format(current_language=state["current_language"], blog_content=blog_content))

        ]
        transaltion_content = self.llm.with_structured_output(Blog).invoke(messages)
        return {"blog": {"content": transaltion_content}}

    def route(self, state: BlogState):
        return {"current_language": state['current_language']}

    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """
        if state["current_language"] == "spanish":
            return "spanish"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state['current_language']


# Setup Graphs:
class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)
        self.blog_node_obj = BlogNode(self.llm)
        print(f"LLM used: {self.llm}")

    def build_topic_graph(self):
        """
        Build a graph to generate blogs based on topic
        """
        # Nodes
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation", self.blog_node_obj.content_generation)

        # Edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)

        return self.graph

    def build_language_graph(self):
        """
        Build a graph for blog generation with inputs topic and language
        """
        # self.blog_node_obj = BlogNode(self.llm)
        # print(self.llm)
        # Nodes
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation", self.blog_node_obj.content_generation)
        self.graph.add_node("spanish_translation",
                            lambda state: self.blog_node_obj.translation({**state, "current_language": "spanish"}))
        self.graph.add_node("french_translation",
                            lambda state: self.blog_node_obj.translation({**state, "current_language": "french"}))
        self.graph.add_node("route", self.blog_node_obj.route)

        # add edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "route")

        # add conditional edges
        self.graph.add_conditional_edges(
            "route",
            self.blog_node_obj.route_decision,
            {
                "spanish": "spanish_translation",
                "french": "french_translation"
            }
        )
        self.graph.add_edge("spanish_translation", END)
        self.graph.add_edge("french_translation", END)
        return self.graph

    def setup_graph(self, usecase):
        if usecase == "topic":
            self.build_topic_graph()
        if usecase == "language":
            print("Language block")
            self.build_language_graph()

        return self.graph.compile()


# langsmith langgraph studio setup
llm = GroqLLM().get_llm()

# get the graph
graph_builder = GraphBuilder(llm)
# graph = graph_builder.build_topic_graph().compile()
graph = graph_builder.build_language_graph().compile()

