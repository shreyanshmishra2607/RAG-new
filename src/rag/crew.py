from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import RagTool
from .tools.custom_tool import ScrapeAndProcessTool
from datetime import datetime
import os

@CrewBase
class Rag:
    """Rag crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'


    model = os.getenv("MODEL", "ollama/llama3.1")  # Correctly formatted with provider
    api_base = os.getenv("API_BASE", "http://localhost:11434")

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1",  # ✅ No 'ollama/' prefix here!
                "base_url": api_base
            }
        },
        "embedding_model": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "base_url": api_base
            }
        }
    }

    @property
    def rag_tool(self):
        return RagTool(config=self.config, summarize=True)



    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[self.rag_tool],  # 🧠 Injecting the custom RAG here
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        # Collect input URLs and prompt from user at runtime
        urls = input("Enter 3 URLs (separate with commas): ").split(",")
        user_prompt = input("Enter your question: ")

        # Initialize the scrape and RAG processing tool
        scrape_and_rag_tool = ScrapeAndProcessTool(urls=urls, user_prompt=user_prompt)
        merged_topic, scraped_text, user_prompt = scrape_and_rag_tool.scrape_and_process()

        # Inject the merged topic, scraped text, and user prompt into the rag_tool for processing
        self.rag_tool.set_data(scraped_text, user_prompt)

        # Use the dynamically extracted topic for the crew input
        inputs = {
            'topic': merged_topic,
            'current_year': str(datetime.now().year)
        }

        return Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[self.research_task(), self.reporting_task()],
            process=Process.sequential,
            verbose=True,
        )
