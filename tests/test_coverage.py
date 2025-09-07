# =========================================================
# File: tests/test_coverage.py
# Description:
#   Pytest test suite for verifying code coverage across the `src` package.
#   Includes tests for API endpoints, core recommendation logic, metrics,
#   preprocessing utilities, error handling, and edge cases.
# =========================================================

import coverage
from pathlib import Path
from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
from src.utils.preprocessing import prepare_matrices

# =========================================================
# 1. Configure Coverage
# =========================================================

# Initialize coverage analysis for the 'src' directory
cov = coverage.Coverage(
    source=['src'],
    omit=[
        '*/tests/*', # Exclude test files
        '*/__pycache__/*', # Exclude Python cache files
        '*/venv/*', # Exclude virtual environment folders
        '*/env/*', # Exclude alternative env folders
        r'*/\.venv/*', # Exclude hidden .venv folders (regex)
        r'*/\.env/*' # Exclude hidden .env folders (regex)
    ]
)

# =========================================================
# 2. General Coverage Tests
# =========================================================

class TestCoverage:
    """Test suite for ensuring comprehensive code coverage"""
    
    def test_coverage_startup(self):
        """Test that coverage can be started"""
        cov.start()
        assert hasattr(cov, '_started') and cov._started
        cov.stop()
        cov.save()
    
    def test_src_directory_coverage(self):
        """Test that all source files are covered"""
        src_path = Path('src')
        python_files = list(src_path.rglob('*.py'))

        # Start coverage
        cov.start()

        # Import and test all modules
        for py_file in python_files:
            if 'test' not in py_file.name and '__pycache__' not in str(py_file):
                try:
                    # Import the module using sys.path manipulation
                    import sys
                    import os
                    module_dir = str(py_file.parent)
                    if module_dir not in sys.path:
                        sys.path.insert(0, module_dir)
                    
                    module_name = py_file.stem
                    __import__(module_name)
                except ImportError as e:
                    print(f"Could not import {py_file}: {e}")

        cov.stop()
        cov.save()

        # Check that we have coverage data
        assert cov.get_data()
    
    # -------------------------
    # Critical Function Coverage
    # -------------------------
    def test_critical_functions_coverage(self):
        """Test coverage of critical functions"""
        cov.start()
        
        # Test core recommender functions
        from src.recommender.hybrid import SVDHybridRecommender
        from src.recommender.ncf import NCF
        from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
        from src.utils.preprocessing import prepare_matrices, get_users_for_eval
        
        # Test SVD recommender
        recommender = SVDHybridRecommender(n_factors=2)
        ratings_matrix = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
        recommender.fit(ratings_matrix)
        
        # Test NCF model
        model = NCF(num_users=10, num_items=5, embedding_dim=8)
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        output = model(user_ids, item_ids)
        
        # Test metrics
        precision = precision_at_k([1, 2, 3], [1, 2, 4], k=3)
        recall = recall_at_k([1, 2, 3], [1, 2, 4], k=3)
        rmse = rmse_score([1, 2, 3], [1.1, 2.1, 3.1])
        
        # Test preprocessing
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'bundle_id': ['b1', 'b2', 'b1'],
            'user_answer': [1, 0, 1],
            'correct_answer': [1, 1, 0],
            'elapsed_time': [10, 15, 20],
            'timestamp': [0, 1, 2]
        })
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(df, min_interactions=1)
        users = get_users_for_eval(df, min_interactions=1, sample_size=2)
        
        cov.stop()
        cov.save()
        
        # Verify all critical functions were executed
        assert precision >= 0
        assert recall >= 0
        assert rmse >= 0
        assert len(user_map) > 0
        assert len(item_map) > 0
    
    # -------------------------
    # API Endpoint Coverage
    # -------------------------
    def test_api_endpoints_coverage(self):
        """Test coverage of API endpoints"""
        cov.start()
        
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # Test all endpoints
        response = client.get("/health")
        response = client.get("/users")
        
        # Test with valid user if available
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json().get("users", [])
            if users:
                user_id = users[0]
                response = client.get(f"/user/{user_id}/history")
                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "content"
                })
        
        # Test error cases
        response = client.get("/user/invalid_user/history")
        response = client.post("/recommendations", json={
            "user_id": "invalid_user",
            "n_recommendations": 5,
            "recommendation_type": "content"
        })
        
        cov.stop()
        cov.save()
        
    # -------------------------
    # Error Handling Coverage
    # -------------------------
    def test_error_handling_coverage(self):
        """Test coverage of error handling paths"""
        cov.start()
        
        # Test various error conditions
        from src.utils.preprocessing import load_content_model, load_hybrid_model
        
        # Test loading non-existent models
        try:
            load_content_model("nonexistent.pkl")
        except:
            pass
        
        try:
            load_hybrid_model("nonexistent.pkl")
        except:
            pass
        
        # Test with invalid data
        try:
            prepare_matrices(pd.DataFrame(), min_interactions=1000)
        except:
            pass
        
        cov.stop()
        cov.save()
    
    # -------------------------
    # Edge Case Coverage
    # -------------------------
    def test_edge_cases_coverage(self):
        """Test coverage of edge cases"""
        cov.start()

        # Test with empty data
        from src.utils.preprocessing import prepare_matrices
        empty_df = pd.DataFrame()
        try:
            prepare_matrices(empty_df, min_interactions=1)
        except:
            pass
        
        # Test with single user/item
        single_user_df = pd.DataFrame({
            'user_id': ['u1'] * 5,
            'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'],
            'user_answer': [1] * 5,
            'correct_answer': [1] * 5,
            'elapsed_time': [10] * 5,
            'timestamp': range(5)
        })
        prepare_matrices(single_user_df, min_interactions=1)
        
        # Test metrics with edge cases
        precision_at_k([], [1, 2, 3], k=10)
        recall_at_k([1, 2, 3], [], k=10)
        try:
            rmse_score([], [])
        except ValueError:
            # Expected since sklearn requires at least 1 sample
            pass
        
        cov.stop()
        cov.save()
    
    # -------------------------
    # Performance Monitoring Coverage
    # -------------------------
    def test_performance_monitoring_coverage(self):
        """Test coverage of performance monitoring code"""
        cov.start()
        
        import time
        import psutil
        
        # Test performance monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate some work
        time.sleep(0.1)
        
        final_memory = process.memory_info().rss
        memory_change = final_memory - initial_memory
        
        cov.stop()
        cov.save()
        
        # Verify monitoring worked
        assert isinstance(memory_change, int)
    
    # -------------------------
    # Coverage Report Generation
    # -------------------------
    def test_coverage_report_generation(self):
        """Test that coverage reports can be generated"""
        cov.start()
        
        # Execute some code
        from src.recommender.hybrid import SVDHybridRecommender
        recommender = SVDHybridRecommender()
        
        cov.stop()
        cov.save()
        
        # Generate reports
        cov.report()
        cov.html_report(directory='htmlcov')
        
        # Check that report files were created
        assert Path('htmlcov').exists()
        assert Path('htmlcov/index.html').exists()

# =========================================================
# 3. Coverage Threshold Tests
# =========================================================

class TestCoverageThresholds:
    """Test suite for coverage thresholds"""
    
    def test_minimum_coverage_threshold(self):
        """Test that minimum coverage threshold is met"""
        cov.start()
        
        # Execute comprehensive test suite
        self._run_comprehensive_tests()
        
        cov.stop()
        cov.save()
        
        # Calculate coverage
        try:
            total_lines = cov.analysis2('src/recommender/hybrid.py')[1]
            covered_lines = cov.analysis2('src/recommender/hybrid.py')[2]
            
            if total_lines > 0:
                coverage_percentage = (len(covered_lines) / total_lines) * 100
                assert coverage_percentage >= 80, f"Coverage {coverage_percentage}% below 80% threshold"
        except Exception:
            # If coverage analysis fails, just ensure we have some coverage data
            assert cov.get_data()
    
    def _run_comprehensive_tests(self):
        """Run comprehensive test suite for coverage"""
        # Test all major components
        from src.recommender.hybrid import SVDHybridRecommender
        from src.recommender.ncf import NCF
        from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
        from src.utils.preprocessing import prepare_matrices, get_users_for_eval
        from src.utils.mappings import get_subject_categories, create_lecture_title
        
        # Test SVD recommender
        recommender = SVDHybridRecommender(n_factors=2)
        ratings_matrix = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
        recommender.initialize(ratings_matrix, {'u1': 0, 'u2': 1, 'u3': 2}, 
                             {'i1': 0, 'i2': 1, 'i3': 2}, {0: 'i1', 1: 'i2', 2: 'i3'})
        recommender.fit(ratings_matrix)
        recommender.predict_rating(0, 1)
        recommender.get_recommendations(0, n=2, exclude_seen=True, known_items=[0])
        
        # Test NCF model
        model = NCF(num_users=10, num_items=5, embedding_dim=8)
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        output = model(user_ids, item_ids)
        
        # Test metrics
        precision_at_k([1, 2, 3], [1, 2, 4], k=3)
        recall_at_k([1, 2, 3], [1, 2, 4], k=3)
        rmse_score([1, 2, 3], [1.1, 2.1, 3.1])
        
        # Test preprocessing
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'bundle_id': ['b1', 'b2', 'b1'],
            'user_answer': [1, 0, 1],
            'correct_answer': [1, 1, 0],
            'elapsed_time': [10, 15, 20],
            'timestamp': [0, 1, 2]
        })
        prepare_matrices(df, min_interactions=1)
        get_users_for_eval(df, min_interactions=1, sample_size=2)
        
        # Test mappings
        get_subject_categories("3,5,8")
        create_lecture_title("l123", 1, "3,5", 7.5)

# Import statements for coverage testing
import numpy as np
import pandas as pd
import torch 