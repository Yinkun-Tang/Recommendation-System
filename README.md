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
- `documents/`: Complementary materials for viewers' reference.

## Prerequisites

1. Install the latest version of Node.js
2. Create Python virtual environment and install all packages according to ```requirement.txt```
3. Run ```npm install``` under the directory ```./frontend``` to install frontend dependencies
4. Download MovieLens 1M Dataset from [MovieLen | GroupLens](https://grouplens.org/datasets/movielens/)
5. Unzip the downloaded dataset and extract the ```ml-1m``` folder to the directory ```./data/raw```

## Installation

1. Clone or download this repository.

## Usage
*Note: The usage procedure is verified on wsl environment with the assumption of chrome browser installation. Check shell script files for detailed implementations.*

### Streamlit
*Note: Assume evaluation mode will be used. If not, feel free to ignore the first step, and running ```evaluation_data_preprocessing.py``` should not be executed for the second step*
1. For model evaluation purpose, parameter choices and alternative preprocessing or computation strategies can be explored and adjusted in ```evaluation_config.json```
2. When configuration is adjusted and confirmed, run ```data_preprocessing.py``` and ```evaluation_data_preprocessing.py``` under the directory ```./backend```
3. After data preprocessing completes, run ```start_st.sh``` under the root directory and follow hints or instructions on the interface

### React Frontend Framework & FastAPI Backend Server
*Note: Evaluation function is not included for this usage, and user_id is randomly selected each time for different recommenders. Different user_id choices can be explored via refreshing the page to randomly choose a new user_id*
1. Before any operation, run ```data_preprocessing.py``` under the directory ```./backend```
2. Under the root directory, run ```./start_dev.sh``` to launch the frontend interface

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
