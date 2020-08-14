# Elasticsearch meets NetBERT

## System architecture

![System architecture](./-/figures/architecture.png)

## Getting Started

### 1. Setup
The following section lists the requirements in order to start running the project.

This project is based on Docker containers, so ensure to have [Docker](https://docs.docker.com/v17.12/install/) installed on your machine. In addition, your machine should dispose from a working version of Python 3.6 as well as the following packages:
- [bert-serving-client](https://pypi.org/project/bert-serving-client/)
- [elasticsearch](https://pypi.org/project/elasticsearch/)
- [pandas](https://pypi.org/project/pandas/)

These libraries can be installed automatically by running the following command in the *code/* repository:
```bash
pip install -r requirements.txt
```

### 2. Deployment
####  Launch the Docker containers
In order to run the containers, run the following command:
```bash
make install
```
