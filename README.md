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

* Clone the repository
* Copy `.env.example` to `.env`: (bash) `cp .env.example .env`
* Fill in your actual credentials in `.env`

Navigate to the repository root and run the following in a terminal:<br>
* Create a virtual environment: (bash) `make create_environment`
* Install dependencies: (bash) `make requirements`
* Activate the conda environment: (bash) `conda activate ds_code_challenge`

## Run

After activating the conda environment, the code can be accessed and run as follows:<br>  
* Run `notebooks/0.0-main-workflow.ipynb`: assignment submission 
* `notebooks/0.2-swimming-pool-detection.ipynb`: computer vision model training and evaluation. 

