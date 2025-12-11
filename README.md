# Recommender System

Recommendation system, as one of the major track and application area within machine learning, has been a popular topic for research in the field of computer science. This demo project aims to allow user to mimic the video platform browsing recommendation mechanism, while researchers and developers may explore the performance of different recommendation strategies and methodologies. 

This project implements four major recommender paradigms, each encapsulated in a modular class structure:
1. User-Based Collaborative Filtering
2. Item-Based Collaborative Filtering
3. Content-Based Recommendation
4. Hybrid Recommender

For cold-start situations, a popularity-based recommender is used by default.

The system supports an independent evaluation pipeline enabling offline experiments with alternative train/test matrices and metrics.

## Project Structure
- `backend/` : Python backend including preprocessing, recommenders, evaluation.
- `frontend/` : Frontend interface built with Node.js.
- `data/` : Raw and processed dataset storage.

## Prerequisites

1. Install the latest version of Node.js
2. Create Python virtual environment and install all packages according to ```requirement.txt```
3. Run ```npm install``` under the directory ```./frontend``` to install frontend dependencies
4. Download MovieLens 1M Dataset from [MovieLen | GroupLens](https://grouplens.org/datasets/movielens/)
5. Unzip the downloaded dataset and extract the ```ml-1m``` folder to the directory ```./data/raw```

## Installation

1. Clone or download this repository.

## Usage - Interactive Interface

1. Before any operation, run ```data_preprocessing.py``` under the directory ```./backend```
2. Under the root directory, run ```uvicorn backend.fast_api.main:app --reload``` to start the backend server
3. Under the directory ```./frontend``` run command ```npm run dev``` to start the frontend interface

or

1. Before any operation, run ```data_preprocessing.py``` under the directory ```./backend```
2. Under the root directory, run ```./start.sh``` to perform local environment cleaning, dependencies installation, and start both backend server and frontend interface. If the frontend fails to open automatically, please click the address presented in the terminal to enter the interface.

## Usage - Methodologies Evaluation and Experiment

1. Before any operation, run ```data_preprocessing.py``` and ```evaluation_data_preprocessing.py``` under the directory ```./backend```
2. For evaluation purposes, activate ```eval_mode``` for recommenders defined in ```recommenders.py``` under the directory ```./backend```
3. Parameter choices and alternative preprocessing or computation strategies can be explored and adjusted in ```evaluation_config.json```
4. After configuration setup and all required datasets generation, run ```evaluation_pipeline.py``` under the directory ```./backend``` to check configured model(s) performance.

## Contributing


Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
