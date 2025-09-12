# Data-Driven Personalized Educational Content Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive recommendation system for personalized educational content, focusing on TOEIC preparation. This system leverages multiple machine learning approaches to provide tailored learning recommendations.

## Features

- **Multiple Recommendation Strategies**:
  - Content-based filtering using SBERT embeddings
  - Collaborative filtering with SVD matrix factorization
  - Neural Collaborative Filtering (NCF)
  - Hybrid approaches combining the above methods
- **Interactive Dashboard**: Streamlit-based UI for exploring recommendations
- **RESTful API**: FastAPI backend for integration with other services
- **Model Evaluation**: Comprehensive metrics for assessing recommendation quality

## Project Structure

```
project-root/
│
├── api/                                          
│   └── main.py                             # FastAPI application for recommendations
│
├── dashboard/                              # Streamlit dashboard application
│   └── app.py                             # Interactive UI for exploring recommendations
│
├── data/                                   
│   ├── cleaned/                           # Cleaned and processed datasets
│   ├── ednet_kt1_sampler.py               # Helper code for sampling
│   ├── lectures.csv                       # Lecture content data
│   ├── questions.csv                      # Question bank data
│   └── sampled_kt1_logs.csv               # Sampled data
│
├── models/                               # Pre-trained model files 
│   ├── advanced_hybrid_model.pkl           # Advanced hybrid model
│   ├── content_based_model_best.pkl        # Content-based best model
│   ├── content_based_model.pkl             # Content-based model
│   ├── content_similarity.pkl              # Content Similarity
│   ├── ncf_model.pth                       # Neural Collaborative Filtering model
│   ├── svd_hybrid_model_best.pkl           # SVD-based hybrid best model  
│   └── svd_hybrid_model.pkl                # SVD-based hybrid model                          
│            
│
├── notebooks/                            # Jupyter notebooks for analysis & development
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_tf_idf.ipynb                    # Content-based filtering with TF-IDF
│   ├── 03_svd_hybrid.ipynb                # SVD-based hybrid model development
│   ├── 04_model_tuning.ipynb              # Hyperparameter optimization
│   ├── 05_advanced_recommendations.ipynb  # Advanced recommendation techniques
│   └── 06_sbert_ncf_subset_test.ipynb     # Experiments to prove feasibility of models
│
├── src/                                   
│   ├── evaluation/                       # Evaluation metrics & analysis
│   │   └── metrics.py                    # Core evaluation metrics
│   │
│   ├── recommender/                      # Recommendation models
│   │   ├── hybrid.py                     # Hybrid recommendation system
│   │   ├── logic.py                      # Core recommendation logic
│   │   └── ncf.py                        # Neural Collaborative Filtering
│   │   └── schemas.py                    # Pydantic data models for request & response validation  
│   │
│   └── utils/                            # Utility functions
│       ├── preprocessing.py              # Data preprocessing
│       └── mappings.py                   # ID mapping utilities
│
├── tests/                                # Unit & integration tests
├── .gitattributes                       # Git attributes file
├── .gitignore                           # Git ignore file
├── conftest.py                          # Add project root & `src/` to Python path
├── pytest.ini                           # Pytest Configuration file
├── recommendation.log                   # Log file
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation (This file)
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maryamzaman30/Data-Driven-Personalized-Educational-Content-Recommendation-System.git
   cd Data-Driven-Personalized-Educational-Content-Recommendation-System
   ```
> Make sure to clone the repository instead of downloading it. If you download it, the `.pkl` files will be corrupted. Also, ensure that you have Git LFS installed before cloning.

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API Server

1. Start the FastAPI server:
   ```bash
   cd api
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

### Running the Dashboard

1. In a new terminal, navigate to the dashboard directory:
   ```bash
   cd dashboard
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`)

## API Testing

### Available Endpoints

- `GET /health` - Health check
- `GET /users` - List all users
- `GET /user/{user_id}/history` - Get user interaction history e.g. `http://127.0.0.1:8000/user/u101324/history` 
- `POST /recommend` - Get recommendations
  - Parameters:
    - `user_id`: Target user ID (e.g., 'u105425')
    - `method`: Recommendation method ('content', 'collaborative', 'hybrid', 'advanced_hybrid')
    - `n`: Number of recommendations to return

### Testing with Browser

You can test GET endpoints directly in your browser:
- Health check: `http://127.0.0.1:8000/health`
- List users: `http://127.0.0.1:8000/users`

### Testing POST Requests with PowerShell

For the recommendation endpoint, use this PowerShell command:

```powershell
$body = @{
    user_id = "u105425"
    method = "content"
    n = 5
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"

# Display the response
$response
```

## Data Sources

- [EdNet Dataset (KT1 & Contents)](https://github.com/riiid/ednet)

## Test documentations

To read about the testing strategy & test suite for this project, refer to the [README file about tests](tests/README%20tests.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The EdNet dataset provided by Riiid!
<<<<<<< HEAD
<<<<<<< HEAD
- Open-source libraries used in this project
=======
- Open-source libraries used in this project
>>>>>>> 4adf5ff (Updated readme file and added runtime.txt)
=======
- Open-source libraries used in this project
>>>>>>> b3e20d6 (Update README, requirements, and remove recommendation.log/runtime.txt)
