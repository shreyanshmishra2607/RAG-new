from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import RagTool, ScrapeWebsiteTool
from .tools.custom_tool import ScrapeAndProcessTool
from datetime import datetime
import os
import tempfile
import yaml

@CrewBase
class Rag:
    """Rag crew"""

    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'

    model = os.getenv("MODEL", "ollama/llama3.1")
    api_base = os.getenv("API_BASE", "http://localhost:11434")

    # Class-level variables to store dynamic data
    topic = "Python"
    user_question = "What is Python?"
    scraped_content = ""
    current_year = str(datetime.now().year)

    def __init__(self):
        # Load YAML configs as dictionaries
        with open(os.path.join(os.path.dirname(__file__), self.agents_config_path)) as f:
            self.agents_config = yaml.safe_load(f)
        
        with open(os.path.join(os.path.dirname(__file__), self.tasks_config_path)) as f:
            self.tasks_config = yaml.safe_load(f)

    @property
    def rag_config(self):
        return {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.1",
                    "base_url": self.api_base
                }
            },
            "embedding_model": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "base_url": self.api_base
                }
            }
        }

    @property
    def rag_tool(self):
        """Return a configured RagTool instance"""
        tool = RagTool(config=self.rag_config, summarize=True)
        
        # Get the original method
        original_run = tool._run
        
        # Define a wrapper function that correctly handles the kwargs
        def run_wrapper(query: str, **kwargs):
            # Format the kwargs properly according to RagTool's expectations
            formatted_kwargs = {"kwargs": kwargs} if kwargs else {"kwargs": {}}
            return original_run(query=query, **formatted_kwargs)
        
        # Replace the original method
        tool._run = run_wrapper
        
        return tool
    
    @property
    def scrape_tool(self):
        """Return a ScrapeWebsiteTool instance"""
        return ScrapeWebsiteTool()

    @agent
    def researcher(self) -> Agent:
        # Format the agent config with our variables
        researcher_config = dict(self.agents_config['researcher'])
        for key, value in researcher_config.items():
            if isinstance(value, str):
                researcher_config[key] = value.format(
                    topic=self.topic,
                    user_question=self.user_question,
                    current_year=self.current_year
                )
        
        return Agent(
            role=researcher_config["role"],
            goal=researcher_config["goal"],
            backstory=researcher_config["backstory"],
            tools=[self.rag_tool, self.scrape_tool],
            verbose=True,
            llm_config={ 
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),
                "base_url": self.api_base
            }
        )

    @agent
    def reporting_analyst(self) -> Agent:
        # Format the agent config with our variables
        analyst_config = dict(self.agents_config['reporting_analyst'])
        for key, value in analyst_config.items():
            if isinstance(value, str):
                analyst_config[key] = value.format(
                    topic=self.topic,
                    user_question=self.user_question,
                    current_year=self.current_year
                )
        
        return Agent(
            role=analyst_config["role"],
            goal=analyst_config["goal"],
            backstory=analyst_config["backstory"],
            verbose=True,
            llm_config={ 
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),
                "base_url": self.api_base
            }
        )

    @task
    def research_task(self) -> Task:
        # Format the task description and expected output with our variables
        research_config = dict(self.tasks_config['research_task'])
        
        description = research_config["description"].format(
            topic=self.topic,
            user_question=self.user_question,
            scraped_content=self.scraped_content,
            current_year=self.current_year
        )
        
        expected_output = research_config["expected_output"].format(
            topic=self.topic,
            user_question=self.user_question,
            current_year=self.current_year
        )
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.researcher()
        )

    @task
    def reporting_task(self) -> Task:
        # Format the task description and expected output with our variables
        reporting_config = dict(self.tasks_config['reporting_task'])
        
        description = reporting_config["description"].format(
            topic=self.topic,
            user_question=self.user_question,
            current_year=self.current_year
        )
        
        expected_output = reporting_config["expected_output"].format(
            topic=self.topic,
            user_question=self.user_question,
            current_year=self.current_year
        )
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.reporting_analyst(),
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        # Collect input URLs and prompt from user at runtime
        urls = input("Enter 3 URLs (separate with commas): ").split(",")
        user_prompt = input("Enter your question: ")

        # Scrape content from URLs
        scraped_texts = []
        topics = []
        
        for url in urls:
            url = url.strip()
            if url:
                # Extract topic from URL
                try:
                    topic = url.rstrip('/').split('/')[-1].replace('-', ' ').replace('_', ' ')
                    # For Python-specific URLs
                    if "python_intro" in url or "python" in url.lower():
                        topic = "Python"
                    topics.append(topic)
                    
                    # Scrape the website
                    scrape_tool = ScrapeWebsiteTool(website_url=url)
                    content = scrape_tool.run()
                    # Clean the content to remove problematic characters
                    content = ''.join(char for char in content if ord(char) < 128)
                    scraped_texts.append(f"Content from {url}:\n{content}")
                    print(f"Successfully scraped {url}")
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
        
        # Merge extracted topics and texts
        self.topic = "Python" if "python" in user_prompt.lower() else (' '.join(topics) if topics else "Research Topic")
        self.user_question = user_prompt
        
        # Create a well-formatted context from scraped content
        context_info = "\n\n".join(scraped_texts)
        if len(context_info) > 8000:  # Limit to avoid token issues
            context_info = context_info[:8000] + "... (content truncated)"
        
        self.scraped_content = context_info
        print(f"Topic set to: {self.topic}")
        print(f"User question: {self.user_question}")
        print(f"Scraped content length: {len(self.scraped_content)} characters")

        # Create and return the crew
        return Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[self.research_task(), self.reporting_task()],
            process=Process.sequential,
            verbose=True,
            llm_config={ 
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),
                "base_url": self.api_base
            }
        )