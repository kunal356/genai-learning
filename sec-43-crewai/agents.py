import os
from crewai import Agent
from tools import yt_tool

from dotenv import load_dotenv

load_dotenv()


# Creating a senior blog content researcher
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided Yt channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, Machine Learning and GenAI and providing suggestion"
    ),
    tools=[yt_tool],
    allow_delegation=True
)

# Creating a senior blog writer agent with YT tool
blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False

)
