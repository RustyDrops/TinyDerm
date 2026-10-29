"""Launcher script for TinyDerm Lab (Clean Autonomous Retraining)."""

import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import requests

from tinyderm.experiment_db import ExperimentDB
from tinyderm.pipeline import TinyDermPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TinyDerm-Runner")

PORTAL_HOST = "127.0.0.1"
PORTAL_PORT = 8501
PORTAL_URL = f"http://{PORTAL_HOST}:{PORTAL_PORT}"


def wait_for_portal(timeout_sec: int = 30) -> bool:
    """Poll portal until responsive."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(PORTAL_URL, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    logger.info("Initializing TinyDerm Lab...")

    # Step 1: Initialize experiment database
    db_path = Path("runs/experiments.sqlite")
    db = ExperimentDB(db_path)
    logger.info(f"Experiment database verified at {db_path}")

    # Step 2: Start Streamlit portal as subprocess
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/dashboard.py",
        "--server.address",
        PORTAL_HOST,
        "--server.port",
        str(PORTAL_PORT),
        "--server.headless",
        "true",
        "--client.toolbarMode",
        "viewer",
        "--browser.serverAddress",
        PORTAL_HOST,
    ]

    logger.info(f"Starting monitoring portal on {PORTAL_URL}...")
    portal_process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Step 3: Wait until responsive
        logger.info("Waiting for portal to become responsive...")
        if wait_for_portal(timeout_sec=30):
            logger.info(f"Portal is ONLINE at {PORTAL_URL}")
        else:
            logger.warning("Portal took longer than expected to respond, proceeding with pipeline...")

        # Step 4: Open browser automatically
        try:
            logger.info(f"Opening browser at {PORTAL_URL}")
            webbrowser.open(PORTAL_URL)
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")

        # Step 5: Start clean training pipeline worker
        logger.info("Starting autonomous clean training pipeline execution...")
        pipeline = TinyDermPipeline(config_path="config/default.yaml", db_path=db_path, force_clean=True)
        pipeline.run()

        logger.info("Pipeline execution finished. Portal remains active. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Shutting down TinyDerm Lab...")
    finally:
        portal_process.terminate()
        portal_process.wait()


if __name__ == "__main__":
    main()
