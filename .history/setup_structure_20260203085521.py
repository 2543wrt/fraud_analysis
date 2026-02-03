import os
import shutil

def organize_project():
    """Move python files to src directory for better structure"""
    base_dir = os.getcwd()
    src_dir = os.path.join(base_dir, 'src')

    # Create src directory if it doesn't exist
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
        print(f"Created directory: {src_dir}")

    # List of files to move
    files_to_move = [
        'anomaly_detector.py',
        'data_generator.py',
        'feature_extractor.py',
        'fraud_analysis.py',
        'real_time_scorer.py',
        'reporting.py',
        'transaction_graph.py',
        'visualization.py'
    ]

    # Move files
    for filename in files_to_move:
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(src_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to src/")
        else:
            print(f"Note: {filename} not found (already moved?)")

    # Create __init__.py to make it a package
    init_path = os.path.join(src_dir, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            pass
        print("Created src/__init__.py")

if __name__ == "__main__":
    organize_project()