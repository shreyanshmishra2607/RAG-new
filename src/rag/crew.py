from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import RagTool, ScrapeWebsiteTool
from .tools.custom_tool import ScrapeAndProcessTool
from datetime import datetime
import os
import tempfile

@CrewBase
class Rag:
    """Rag crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    model = os.getenv("MODEL", "ollama/llama3.1")
    api_base = os.getenv("API_BASE", "http://localhost:11434")

    rag_config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1",
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
        """Return a configured RagTool instance"""
        tool = RagTool(config=self.rag_config, summarize=True)
        
        # Override the tool's run method to handle kwargs correctly
        original_run = tool._run

        def run_with_kwargs(query: str, **raw_kwargs):
            if not raw_kwargs:
                raw_kwargs = {}
            return original_run(query=query, kwargs=raw_kwargs)

        tool._run = run_with_kwargs
        return tool
    
    @property
    def scrape_tool(self):
        """Return a ScrapeWebsiteTool instance"""
        return ScrapeWebsiteTool()

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
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
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm_config={ 
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),
                "base_url": self.api_base
            }
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

        # Instead of trying to set_data which doesn't exist,
        # we'll pass the information directly as context
        
        # First, scrape content from URLs
        scraped_texts = []
        topics = []
        
        for url in urls:
            url = url.strip()
            if url:
                # Extract topic from URL
                topic = url.rstrip('/').split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                topics.append(topic)
                
                # Scrape the website
                try:
                    scrape_tool = ScrapeWebsiteTool(website_url=url)
                    content = scrape_tool.run()
                    # Clean the content to remove problematic characters
                    content = ''.join(char for char in content if ord(char) < 128)
                    scraped_texts.append(content)
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
        
        # Merge extracted topics and texts
        merged_topic = ' '.join(topics) if topics else "Research Topic"
        context_info = f"User Question: {user_prompt}\n\nScraped Content Summary:\n"
        context_info += "\n\n".join(scraped_texts[:1000])  # Limit the size to avoid encoding issues
                
        # Set up the inputs for the crew
        inputs = {
            'topic': merged_topic,
            'user_question': user_prompt,
            'scraped_content': context_info,
            'current_year': str(datetime.now().year)
        }

        return Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[self.research_task(), self.reporting_task()],
            process=Process.sequential,
            verbose=True,
            inputs=inputs,
            llm_config={ 
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),
                "base_url": self.api_base
            }
        )