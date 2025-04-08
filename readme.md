# Intro to Vector Databases

This project demonstrates the use of vector databases for storing and retrieving information.

## Description

This project includes the following files:

- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `ingestion.py`: Script for ingesting and processing text data for vector storage.
- `mediumblog.txt`: Text data containing information about embeddings and vector databases.
- `Pipfile`: Specifies the project dependencies.
- `retrievalvectordb.py`: Script for retrieving data from the vector database.
- `vectordb.py`: Script for creating and interacting with a Chroma vector database.

## Installation

1. Install the dependencies using Pipenv:

   ```bash
   pipenv install
   ```

## Usage

1. Run the `ingestion.py` script to ingest the data into the vector database:

   ```bash
   python ingestion.py
   ```

2. Run the `retrievalvectordb.py` script to retrieve data from the vector database:

   ```bash
   python retrievalvectordb.py
   ```

## Vector Databases

This project uses Chroma as the vector database.

## Embeddings

This project uses OpenAI embeddings.