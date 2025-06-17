# Data-Driven Personalized Educational Content Recommendation System

view original EdNet dataset **KT1 & Contents** [here](https://github.com/riiid/ednet)

## Project Structure

```
project-root/
│
├── data/                
│   ├── cleaned/                            # Cleaned and processed datasets
├── src/                                          
│   ├── evaluation/                         # Evaluation metrics and analysis
│   │   └── metrics.py                      # Core evaluation metrics
│   ├── recommender/                        # Recommendation models
│   │   └── hybrid.py                       # Hybrid recommendation system
│   └── utils/                              # Utility functions
│       ├── preprocessing.py                # Data preprocessing
│       └── mappings.py                     # ID mapping utilities
│                                                    
├── notebooks/                                         
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_tf_idf.ipynb                     # Minimal vial product (MVP)
│   ├── 03_svd_hybrid.ipynb                 # Minimal vial product (MVP)
│   ├── 04_model_tuning.ipynb               # Optimizes hybrid model weights
│   └── 05_advanced_recommendations.ipynb   # Integrates NCF & BERT
│                                                
├── api/                                          
│   └── main.py                             # Main API application
│                                                  
├── models/                                 # Trained model files
│                                                   
├── dashboard/                              # Streamlit dashboard application
│   └── app.py                                            
│                                                   
├── tests/                                  # Unit tests          
│                                                  
├── requirements.txt                        # Project dependencies
└── README.md                               # Project Setup & documentation
```

## Development Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```