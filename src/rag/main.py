#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from rag.crew import Rag

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    try:
        # Get the dynamically passed topic from crew.py
        topic = Rag().crew().topic  # This will come from the ScrapeAndProcessTool
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
        # Get the dynamically passed topic from crew.py
        topic = Rag().crew().topic  # This will come from the ScrapeAndProcessTool
        inputs = {
            'topic': topic,
        }
        Rag().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Rag().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and return the results.
    """
    try:
        # Get the dynamically passed topic from crew.py
        topic = Rag().crew().topic  # This will come from the ScrapeAndProcessTool
        inputs = {
            'topic': topic,
            'current_year': str(datetime.now().year)
        }
        Rag().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
