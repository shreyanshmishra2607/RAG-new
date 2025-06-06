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
        # Simply create and run the crew
        result = Rag().crew().kickoff()
        print("Final result:", result)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check that Ollama is running at http://localhost:11434 and has the required models.")

if __name__ == "__main__":
    run()