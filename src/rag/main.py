#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from rag.crew import Rag
from rag.tools import ScrapeAndProcessTool  # Assuming this is the custom tool you're using

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    try:
        # Get the dynamically passed topic from ScrapeAndProcessTool
        topic = ScrapeAndProcessTool().scrape_topic()  # Assuming method scrape_topic() fetches the topic
        inputs = {
            'topic': topic,  # Use the dynamically set topic here
            'current_year': str(datetime.now().year)
        }
        Rag().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        # Get the dynamically passed topic from ScrapeAndProcessTool
        topic = ScrapeAndProcessTool().scrape_topic()  # Assuming method scrape_topic() fetches the topic
        inputs = {
            'topic': topic,
        }
        if len(sys.argv) != 3:
            raise ValueError("Please provide the number of iterations and filename as arguments.")
        Rag().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide the task ID as an argument.")
        Rag().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and return the results.
    """
    try:
        # Get the dynamically passed topic from ScrapeAndProcessTool
        topic = ScrapeAndProcessTool().scrape_topic()  # Assuming method scrape_topic() fetches the topic
        inputs = {
            'topic': topic,
            'current_year': str(datetime.now().year)
        }
        if len(sys.argv) != 3:
            raise ValueError("Please provide the number of iterations and model name as arguments.")
        Rag().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    # Add a check for the operation you want to perform: run, train, replay, or test
    if len(sys.argv) < 2:
        print("Usage: python main.py <operation> <additional_args>")
        sys.exit(1)

    operation = sys.argv[1].lower()

    if operation == "run":
        run()
    elif operation == "train":
        train()
    elif operation == "replay":
        replay()
    elif operation == "test":
        test()
    else:
        print(f"Unknown operation: {operation}")
        sys.exit(1)
