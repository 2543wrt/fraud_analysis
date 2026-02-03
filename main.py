import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

from src import fraud_analysis

if __name__ == "__main__":
    # Run the main analysis function
    fraud_analysis.main()