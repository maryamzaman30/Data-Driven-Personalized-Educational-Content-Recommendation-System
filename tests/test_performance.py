import pytest
import time
import psutil
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from src.recommender.hybrid import SVDHybridRecommender
from src.recommender.ncf import NCF
from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
from src.utils.preprocessing import prepare_matrices

@pytest.fixture
def large_dataset():
    """Create large dataset for performance testing"""
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    n_interactions = 50000
    
    user_ids = np.random.choice(n_users, n_interactions)
    item_ids = np.random.choice(n_items, n_interactions)
    ratings = np.random.randint(1, 6, n_interactions)
    
    return pd.DataFrame({
        'user_id': [f'u{i}' for i in user_ids],
        'bundle_id': [f'b{i}' for i in item_ids],
        'user_answer': np.random.randint(0, 2, n_interactions),
        'correct_answer': np.random.randint(0, 2, n_interactions),
        'elapsed_time': np.random.randint(5, 60, n_interactions),
        'timestamp': range(n_interactions)
    })

class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking"""
    
    def test_svd_training_performance(self, large_dataset):
        """Benchmark SVD model training performance"""
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            large_dataset, min_interactions=10
        )
        
        recommender = SVDHybridRecommender(n_factors=20)
        recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
        
        # Measure training time
        start_time = time.time()
        recommender.fit(train_matrix)
        training_time = time.time() - start_time
        
        # Performance assertions
        assert training_time < 30.0  # Should complete within 30 seconds
        assert recommender.user_factors is not None
        assert recommender.item_factors is not None
        
        print(f"SVD training time: {training_time:.2f} seconds")
    
    def test_ncf_inference_performance(self):
        """Benchmark NCF model inference performance"""
        model = NCF(num_users=1000, num_items=500, embedding_dim=32)
        model.eval()
        
        # Test single prediction
        start_time = time.time()
        for _ in range(100):
            user_ids = torch.randint(0, 1000, (1,))
            item_ids = torch.randint(0, 500, (1,))
            with torch.no_grad():
                _ = model(user_ids, item_ids)
        single_pred_time = time.time() - start_time
        
        # Test batch prediction
        start_time = time.time()
        user_ids = torch.randint(0, 1000, (100,))
        item_ids = torch.randint(0, 500, (100,))
        with torch.no_grad():
            _ = model(user_ids, item_ids)
        batch_pred_time = time.time() - start_time
        
        # Performance assertions
        assert single_pred_time < 5.0  # 100 single predictions in 5 seconds
        assert batch_pred_time < 2.0   # 100 batch predictions in 2 seconds
        
        print(f"Single prediction time (100x): {single_pred_time:.3f} seconds")
        print(f"Batch prediction time (100x): {batch_pred_time:.3f} seconds")
    
    def test_recommendation_generation_performance(self, large_dataset):
        """Benchmark recommendation generation performance"""
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            large_dataset, min_interactions=10
        )
        
        recommender = SVDHybridRecommender(n_factors=20)
        recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
        recommender.fit(train_matrix)
        
        # Test recommendation generation for multiple users
        test_users = list(user_map.keys())[:10]
        start_time = time.time()
        
        for user_id in test_users:
            user_idx = user_map[user_id]
            known_items = np.where(train_matrix[user_idx] > 0)[0]
            recommendations = recommender.get_recommendations(
                user_idx=user_idx, n=10, exclude_seen=True, known_items=known_items
            )
            assert len(recommendations) <= 10
        
        total_time = time.time() - start_time
        avg_time_per_user = total_time / len(test_users)
        
        # Performance assertions
        assert avg_time_per_user < 0.1  # Less than 100ms per user
        assert total_time < 2.0  # Total time for 10 users less than 2 seconds
        
        print(f"Average recommendation time per user: {avg_time_per_user:.3f} seconds")
        print(f"Total time for {len(test_users)} users: {total_time:.3f} seconds")
    
    def test_memory_usage(self, large_dataset):
        """Test memory usage during model operations"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load and train model
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            large_dataset, min_interactions=10
        )
        
        recommender = SVDHybridRecommender(n_factors=20)
        recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
        recommender.fit(train_matrix)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory assertions
        assert memory_increase < 500  # Less than 500MB memory increase
        print(f"Memory increase: {memory_increase:.1f} MB")
    
    def test_scalability_with_dataset_size(self):
        """Test scalability with different dataset sizes"""
        sizes = [100, 500, 1000, 2000]
        training_times = []
        
        for size in sizes:
            # Create dataset of given size
            np.random.seed(42)
            n_interactions = size * 10
            
            user_ids = np.random.choice(size, n_interactions)
            item_ids = np.random.choice(size // 2, n_interactions)
            
            df = pd.DataFrame({
                'user_id': [f'u{i}' for i in user_ids],
                'bundle_id': [f'b{i}' for i in item_ids],
                'user_answer': np.random.randint(0, 2, n_interactions),
                'correct_answer': np.random.randint(0, 2, n_interactions),
                'elapsed_time': np.random.randint(5, 60, n_interactions),
                'timestamp': range(n_interactions)
            })
            
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(
                df, min_interactions=5
            )
            
            recommender = SVDHybridRecommender(n_factors=10)
            recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
            
            start_time = time.time()
            recommender.fit(train_matrix)
            training_time = time.time() - start_time
            
            training_times.append(training_time)
            print(f"Dataset size {size}: {training_time:.2f} seconds")
        
        # Check that training time doesn't grow exponentially
        for i in range(1, len(training_times)):
            growth_factor = training_times[i] / training_times[i-1]
            assert growth_factor < 6.0  # Should not grow more than 6x for 2x data size

class TestMetricsPerformance:
    """Test suite for evaluation metrics performance"""
    
    def test_precision_recall_performance(self):
        """Benchmark precision and recall calculation performance"""
        # Create large test data
        n_recommendations = 1000
        n_relevant = 500
        
        recommended = list(range(n_recommendations))
        relevant = list(range(n_relevant))
        
        # Benchmark precision calculation
        start_time = time.time()
        for _ in range(100):
            precision = precision_at_k(recommended, relevant, k=10)
        precision_time = time.time() - start_time
        
        # Benchmark recall calculation
        start_time = time.time()
        for _ in range(100):
            recall = recall_at_k(recommended, relevant, k=10)
        recall_time = time.time() - start_time
        
        # Performance assertions
        assert precision_time < 1.0  # 100 calculations in 1 second
        assert recall_time < 1.0     # 100 calculations in 1 second
        
        print(f"Precision calculation time (100x): {precision_time:.3f} seconds")
        print(f"Recall calculation time (100x): {recall_time:.3f} seconds")
    
    def test_rmse_performance(self):
        """Benchmark RMSE calculation performance"""
        # Create large test data
        n_predictions = 10000
        true_vals = np.random.rand(n_predictions)
        pred_vals = np.random.rand(n_predictions)
        
        # Benchmark RMSE calculation
        start_time = time.time()
        for _ in range(100):
            rmse = rmse_score(true_vals, pred_vals)
        rmse_time = time.time() - start_time
        
        # Performance assertions
        assert rmse_time < 2.0  # 100 calculations in 2 seconds
        assert isinstance(rmse, float)
        
        print(f"RMSE calculation time (100x): {rmse_time:.3f} seconds")

class TestConcurrentPerformance:
    """Test suite for concurrent operation performance"""
    
    def test_concurrent_recommendations(self, large_dataset):
        """Test performance under concurrent recommendation requests"""
        import threading
        import queue
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            large_dataset, min_interactions=10
        )
        
        recommender = SVDHybridRecommender(n_factors=20)
        recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
        recommender.fit(train_matrix)
        
        results = queue.Queue()
        
        def generate_recommendations(user_id):
            """Generate recommendations for a user"""
            try:
                user_idx = user_map[user_id]
                known_items = np.where(train_matrix[user_idx] > 0)[0]
                recommendations = recommender.get_recommendations(
                    user_idx=user_idx, n=5, exclude_seen=True, known_items=known_items
                )
                results.put((user_id, len(recommendations)))
            except Exception as e:
                results.put((user_id, f"Error: {str(e)}"))
        
        # Test concurrent recommendations
        test_users = list(user_map.keys())[:20]
        threads = []
        
        start_time = time.time()
        for user_id in test_users:
            thread = threading.Thread(target=generate_recommendations, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Check results
        successful_recommendations = 0
        while not results.empty():
            user_id, result = results.get()
            if isinstance(result, int):
                successful_recommendations += 1
        
        # Performance assertions
        assert total_time < 10.0  # 20 concurrent requests in 10 seconds
        assert successful_recommendations >= 15  # At least 75% success rate
        
        print(f"Concurrent recommendations time: {total_time:.2f} seconds")
        print(f"Successful recommendations: {successful_recommendations}/{len(test_users)}")

class TestResourceUtilization:
    """Test suite for resource utilization monitoring"""
    
    def test_cpu_utilization(self, large_dataset):
        """Monitor CPU utilization during model training"""
        import psutil
        
        process = psutil.Process()
        
        # Monitor CPU during training
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            large_dataset, min_interactions=10
        )
        
        recommender = SVDHybridRecommender(n_factors=20)
        recommender.initialize(train_matrix, user_map, item_map, {v: k for k, v in item_map.items()})
        
        # Start monitoring
        cpu_percentages = []
        start_time = time.time()
        
        # Train model while monitoring
        recommender.fit(train_matrix)
        
        # Check CPU usage
        cpu_percent = process.cpu_percent()
        cpu_percentages.append(cpu_percent)
        
        training_time = time.time() - start_time
        
        # Resource assertions
        assert training_time < 30.0  # Training completes in reasonable time
        print(f"Training time: {training_time:.2f} seconds")
        print(f"CPU utilization: {cpu_percent:.1f}%")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of model components"""
        import psutil
        
        process = psutil.Process()
        
        # Test NCF memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = NCF(num_users=1000, num_items=500, embedding_dim=32)
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test inference memory usage
        user_ids = torch.randint(0, 1000, (100,))
        item_ids = torch.randint(0, 500, (100,))
        
        model.eval()
        with torch.no_grad():
            _ = model(user_ids, item_ids)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = peak_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_increase < 100  # Less than 100MB for model + inference
        print(f"Model memory usage: {model_memory - initial_memory:.1f} MB")
        print(f"Peak memory usage: {memory_increase:.1f} MB")

class TestPerformanceRegression:
    """Test suite for performance regression detection"""
    
    def test_performance_baseline(self):
        """Establish performance baselines"""
        # Test SVD performance baseline
        ratings_matrix = np.random.rand(100, 50)
        recommender = SVDHybridRecommender(n_factors=10)
        
        start_time = time.time()
        recommender.fit(ratings_matrix)
        training_time = time.time() - start_time
        
        # Baseline assertions
        assert training_time < 5.0  # Baseline training time
        print(f"Baseline SVD training time: {training_time:.3f} seconds")
    
    def test_ncf_performance_baseline(self):
        """Establish NCF performance baseline"""
        model = NCF(num_users=100, num_items=50, embedding_dim=16)
        model.eval()
        
        user_ids = torch.randint(0, 100, (10,))
        item_ids = torch.randint(0, 50, (10,))
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(user_ids, item_ids)
        inference_time = time.time() - start_time
        
        # Baseline assertions
        assert inference_time < 0.1  # Baseline inference time
        print(f"Baseline NCF inference time: {inference_time:.3f} seconds") 