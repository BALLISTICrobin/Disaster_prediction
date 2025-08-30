# Truly Predictive Natural Disaster API

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-blue)

![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-green)

This project provides a high-performance REST API that leverages a machine learning model to predict the probability of different types of natural disasters (specifically Flood, Storm, Landslide, and Wildfire) based on geographical location and time of year.

## Features

***High-Performance API**: Built with FastAPI for fast and asynchronous request handling.

***Accurate Predictions**: Utilizes a fine-tuned `XGBoost` classifier for robust and accurate probability predictions.

***Intelligent Feature Engineering**: Converts dates (month and day) into cyclical features (`sin`/`cos`) to capture seasonal patterns effectively.

***Ready for Production**: Includes model training and serving logic, with clear separation of concerns.

***Interactive Documentation**: Automatic generation of interactive API docs via Swagger UI and ReDoc.

## Tech Stack

***Backend**: Python, FastAPI

***ML/Data Science**: XGBoost, Scikit-learn, Pandas, NumPy

***Server**: Uvicorn

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.9 or higher
* Git
* The disaster dataset. You must have the `1970-2021_DISASTERS.xlsx - emdat data.csv` file in the project's root directory. This file is used by `train_model.py`.

### Installation

1.**Clone the repository:**

    ```bash

git clone `<your-repository-url>`

cd FastApiForNaturalDisater

    ```

2.**Create and activate a virtual environment:**

***On macOS/Linux:**

    ```bash

python3 -m venv venv

source venv/bin/activate

    ```

***On Windows:**

    ```bash

python -m venv venv

.\venv\Scripts\activate

    ```

3.**Install the required dependencies:**

    ```bash

pip install -r requirements.txt

    ```

## Usage

The application requires a two-step process to run: first, you train the model, and then you run the API server.

### 1. Train the Predictive Model

Before you can run the API, you must first train the machine learning model. The training script processes the raw data, engineers features, tunes hyperparameters, and saves the final model and data encoders.

Execute the training script from the root directory:

```bash

pythontrain_model.py

```

This will generate the following files in your project directory:

*`disaster_model.json` (the trained XGBoost model)

*`country_encoder.joblib`

*`region_encoder.joblib`

*`continent_encoder.joblib`

*`disaster_encoder.joblib`

### 2. Run the API Server

Once the model and encoders are saved, you can start the FastAPI server using Uvicorn.

```bash

uvicornmain:app--reload

```

The server will be available at `http://127.0.0.1:8000`. The `--reload` flag enables hot-reloading for development.

### 3. Access Interactive API Docs

FastAPI provides automatically generated documentation. Once the server is running, you can access it at:

***Swagger UI**: `http://127.0.0.1:8000/docs`

***ReDoc**: `http://127.0.0.1:8000/redoc`

---

## API Endpoint

### POST `/predict/risk`

Calculates the probability of various natural disasters for a given location and date.

**Request Body:**

```json

{

"year":2025,

"country":"Bangladesh",

"region":"Southern Asia",

"continent":"Asia",

"month":9,

"day":29

}

```

**Example `curl` Request:**

```bash

curl-X'POST'\

  'http://127.0.0.1:8000/predict/risk' \

-H'accept: application/json'\

  -H 'Content-Type: application/json' \

-d'{

    "year": 2025,

    "country": "Bangladesh",

    "region": "Southern Asia",

    "continent": "Asia",

    "month": 9,

    "day": 29

  }'

```

**Example Success Response (200 OK):**

The response contains the predicted probability for each disaster type.

```json

{

"Flood":"45.13%",

"Landslide":"5.67%",

"Storm":"48.98%",

"Wildfire":"0.22%"

}

```

**Example Error Response (404 Not Found):**

If a provided location (country, region, or continent) is not found in the training data, the API will return a 404 error.

```json

{

"detail":"Country 'Atlantis' not found."

}

```

## Project Structure

```

.

├── main.py             # FastAPI application logic and prediction endpoint

├── train_model.py      # Script to train the ML model and encoders

├── requirements.txt    # Project dependencies

├── README.md           # This file

├── 1970-2021_DISASTERS.xlsx - emdat data.csv # (Required) Training data

├── disaster_model.json # (Generated) Trained XGBoost model

└── *.joblib            # (Generated) Saved data encoders

```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
