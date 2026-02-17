import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sql_db import init_db
from vector_store import ingest_data

def main():
    print("Initializing SQL Database...")
    init_db()
    print("SQL Database Initialized.")

    print("Ingesting Vector Data...")
    ingest_data()
    print("Vector Data Ingested.")

if __name__ == "__main__":
    main()
