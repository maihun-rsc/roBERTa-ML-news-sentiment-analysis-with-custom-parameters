import os
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Ordered list of modules/files to commit day by day
COMMITS = [
    {
        "message": "Day 2: Implement RSS data collection and scraping logic",
        "files": ["data_collection/*.py"]
    },
    {
        "message": "Day 3: Add text preprocessing and feature extraction modules",
        "files": ["preprocessing/"]
    },
    {
        "message": "Day 4: Implement supervised ML models for sentiment analysis",
        "files": ["models/"]
    },
    {
        "message": "Day 5: Add analysis and evaluation reporting scripts",
        "files": ["analysis/"]
    },
    {
        "message": "Day 6: Main orchestration script and testing pipelines",
        "files": ["main.py", "test_pipeline.py", "requirements.txt"]
    }
]

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
        logging.info(f"Command successful: {cmd}\n{res.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}\n{e.stderr}")
        return False

def main():
    logging.info("Starting auto-commit script. Will push one module every 24 hours.")
    
    for i, commit in enumerate(COMMITS):
        # Wait 24 hours (86400 seconds) before each commit
        wait_time = 86400
        logging.info(f"Sleeping for 24 hours before pushing '{commit['message']}'...")
        time.sleep(wait_time)
        
        # Add files
        files_to_add = " ".join(commit["files"])
        logging.info(f"Adding files: {files_to_add}")
        run_cmd(f"git add {files_to_add}")
        
        # Commit
        logging.info(f"Committing: {commit['message']}")
        run_cmd(f'git commit -m "{commit["message"]}"')
        
        # Push
        logging.info("Pushing to origin main")
        run_cmd("git push origin main")
        
    logging.info("All modules have been pushed!")

if __name__ == "__main__":
    main()
