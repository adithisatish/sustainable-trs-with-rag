---
license: mit
language:
- en
---
# Enhancing Sustainable Travel Recommendations using Retrieval-Augmented Generation

In this project, we use RAGs to enhance the recommendation capability of LLMs by curating a knowledge base of tourism information for European cities, along with relevant sustainability information. 

Steps to run the code locally: (add detailed steps - TODO)

1. Set up the a virtual environment, install and log in to HuggingFace-Client and install the requirements
2. Request access to the models
3. Clone the data repository from HuggingFace
4. Set up the database 
5. Run the pipeline

### Points to remember during project setup
- the vector `database` is created under `sustainable-trs-with-rag/database/`
- the data from HF repo is stored under `sustainable-trs-with-rag/european-city-data/`
- To run the single pipeline from the command line, use the following command:
```python pipeline.py``` from the `src` directory.
- If you are using an IDE e.g. Pycharm, you can run the `pipeline.py` file directly by making sure you set the src directory as your `"Sources Root"`.
### Directory Structure
```
|-- README.md
|-- database
|-- european-city-data
|-- requirements.txt
|-- src
|   |-- __init__.py
|   |-- augmentation
|   |   |-- __init__.py
|   |   |-- prompt_generation.py
|   |-- data_directories.py
|   |-- information_retrieval
|   |   |-- __init__.py
|   |   |-- info_retrieval.py
|   |-- pipeline.py
|   |-- sustainability
|   |   |-- __init__.py
|   |   |-- s_fairness.py
|   |-- text_generation
|   |   |-- __init__.py
|   |   |-- llm_test.ipynb
|   |   |-- model_init.py
|   |   |-- text_generation.py
|   |-- vectordb
|       |-- __init__.py
|       |-- create_db.py
|       |-- helpers.py
|       |-- lancedb_init.py
|       |-- vectordb.py
|-- tests
```
