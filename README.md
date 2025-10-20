# DS Code Challenge

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

DS coding challenge for CoCT

## Prerequisites
You need the following installed to run this code:
- conda
- make

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env`: (bash) `cp .env.example .env`
3. Fill in your actual credentials in `.env`

Navigate to the repository root and run the following in a terminal:
4. Create a virtual environment: (bash) `make create_environment`
5. Install dependencies: (bash) `make requirements`
6. Activate the conda environment: (bash) `conda activate ds_code_challenge`

## Run

After activating the conda environment, the code can be accessed and run as follows: 
7. Open and run `notebooks/0.0-main-workflow.ipynb` for assignment submission
8. Go to `notebooks/0.2-swimming-pool-detection.ipynb` for computer vision model training and evaluation. 

