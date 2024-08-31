from crewai import Crew,Process
from agents import blog_researcher, blog_writer
from tasks import research_task, write_task

crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result = crew.kickoff(inputs={'topic': 'AI VS ML VS DL vs Data Science'})
print(result)