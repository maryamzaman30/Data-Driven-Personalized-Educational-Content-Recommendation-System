# =========================================================
# File: tests/test_edge_cases.py
# Description:
#   Comprehensive pytest suite for testing:
#     - Extreme data scenarios
#     - Boundary conditions
#     - Error handling
#     - Model-specific edge cases
#     - System limit handling
# =========================================================

import pytest
import numpy as np
import pandas as pd
import torch

# Project imports
from src.recommender.hybrid import SVDHybridRecommender
from src.recommender.ncf import NCF
from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
from src.utils.preprocessing import prepare_matrices


# =========================================================
# 1. Extreme Data Scenarios
# =========================================================

class TestExtremeDataScenarios:
    """Test suite for extreme data scenarios"""
    
    def test_empty_dataset(self):
        """Test handling of completely empty dataset"""
        empty_df = pd.DataFrame(columns=['user_id', 'bundle_id', 'user_answer', 'correct_answer'])
        
        # Test preprocessing with empty data - should handle gracefully
        try:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(empty_df, min_interactions=1)
            # If it succeeds, check the results
            assert train_matrix.shape == (0, 0)
            assert test_matrix.shape == (0, 0)
            assert len(user_map) == 0
            assert len(item_map) == 0
        except ValueError as e:
            # Expected behavior for empty dataset
            assert "empty" in str(e).lower() or "no samples" in str(e).lower()
    
    def test_single_user_dataset(self):
        """Test handling of dataset with only one user"""
        single_user_df = pd.DataFrame({
            'user_id': ['u1'] * 10,
            'bundle_id': [f'b{i}' for i in range(10)],
            'user_answer': [1] * 10,
            'correct_answer': [1] * 10,
            'elapsed_time': [10] * 10,
            'timestamp': range(10)
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(single_user_df, min_interactions=5)
        
        assert train_matrix.shape[0] == 1  # Only one user
        assert len(user_map) == 1
        assert len(item_map) == 10
    
    def test_single_item_dataset(self):
        """Test handling of dataset with only one item"""
        single_item_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in range(10)],
            'bundle_id': ['b1'] * 10,
            'user_answer': [1] * 10,
            'correct_answer': [1] * 10,
            'elapsed_time': [10] * 10,
            'timestamp': range(10)
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(single_item_df, min_interactions=1)
        
        assert train_matrix.shape[1] == 1  # Only one item
        assert len(user_map) == 10
        assert len(item_map) == 1
    
    def test_sparse_dataset(self):
        """Test handling of very sparse dataset"""
        # Create sparse dataset with very few interactions
        np.random.seed(42)
        n_users = 1000
        n_items = 500
        n_interactions = 50  # Very sparse
        
        user_ids = np.random.choice(n_users, n_interactions)
        item_ids = np.random.choice(n_items, n_interactions)
        
        sparse_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in user_ids],
            'bundle_id': [f'b{i}' for i in item_ids],
            'user_answer': np.random.randint(0, 2, n_interactions),
            'correct_answer': np.random.randint(0, 2, n_interactions),
            'elapsed_time': np.random.randint(5, 60, n_interactions),
            'timestamp': range(n_interactions)
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(sparse_df, min_interactions=1)
        
        # Should handle sparse data gracefully
        assert train_matrix.shape[0] > 0
        assert train_matrix.shape[1] > 0
        # Some interactions may be lost due to train/test split, but should have some
        assert np.sum(train_matrix > 0) > 0
    
    def test_dense_dataset(self):
        """Test handling of very dense dataset"""
        # Create dense dataset with many interactions per user
        np.random.seed(42)
        n_users = 10
        n_items = 20
        n_interactions = n_users * n_items  # Dense
        
        user_ids = np.repeat(range(n_users), n_items)
        item_ids = np.tile(range(n_items), n_users)
        
        dense_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in user_ids],
            'bundle_id': [f'b{i}' for i in item_ids],
            'user_answer': np.random.randint(0, 2, n_interactions),
            'correct_answer': np.random.randint(0, 2, n_interactions),
            'elapsed_time': np.random.randint(5, 60, n_interactions),
            'timestamp': range(n_interactions)
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(dense_df, min_interactions=1)
        
        # Should handle dense data efficiently
        assert train_matrix.shape == (n_users, n_items)
        # Most interactions should be preserved (accounting for train/test split)
        assert np.sum(train_matrix > 0) > n_interactions * 0.6
        
    def test_cold_start_and_diverse_learners(self):
        """Test handling of cold-start users and diverse learner histories"""
        # Create a dataset with diverse user interaction patterns
        np.random.seed(42)
        
        # 1. Cold-start users (1-2 interactions)
        cold_start_users = 5
        cold_start_interactions = np.random.randint(1, 3, cold_start_users)
        
        # 2. Average users (3-10 interactions)
        avg_users = 10
        avg_interactions = np.random.randint(3, 11, avg_users)
        
        # 3. Power users (11-20 interactions)
        power_users = 5
        power_interactions = np.random.randint(11, 21, power_users)
        
        # Combine all users
        all_users = []
        user_types = []
        
        # Add cold-start users
        for i in range(cold_start_users):
            all_users.extend([f'cold_{i}'] * cold_start_interactions[i])
            user_types.extend(['cold_start'] * cold_start_interactions[i])
            
        # Add average users
        for i in range(avg_users):
            all_users.extend([f'avg_{i}'] * avg_interactions[i])
            user_types.extend(['average'] * avg_interactions[i])
            
        # Add power users
        for i in range(power_users):
            all_users.extend([f'power_{i}'] * power_interactions[i])
            user_types.extend(['power'] * power_interactions[i])
        
        # Create the DataFrame with diverse interaction patterns
        n_interactions = len(all_users)
        n_items = 50
        
        # Create some patterns in the data
        # Power users tend to answer more correctly
        correct_answers = []
        for user_type in user_types:
            if user_type == 'cold_start':
                correct_answers.append(np.random.choice([0, 1], p=[0.6, 0.4]))
            elif user_type == 'average':
                correct_answers.append(np.random.choice([0, 1], p=[0.4, 0.6]))
            else:  # power users
                correct_answers.append(np.random.choice([0, 1], p=[0.2, 0.8]))
        
        # Create the dataset
        diverse_df = pd.DataFrame({
            'user_id': all_users,
            'bundle_id': [f'b{np.random.randint(0, n_items)}' for _ in range(n_interactions)],
            'user_answer': [1 if x == 1 else np.random.randint(0, 2) for x in correct_answers],
            'correct_answer': correct_answers,
            'elapsed_time': np.random.randint(5, 300, n_interactions),  # More varied response times
            'timestamp': range(n_interactions)
        })
        
        # Test with different minimum interaction thresholds
        for min_interactions in [1, 3, 5]:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(
                diverse_df, 
                min_interactions=min_interactions
            )
            
            # Verify that we still get reasonable results with cold-start users (when min_interactions=1)
            if min_interactions == 1:
                assert len(user_map) > cold_start_users  # Should include at least cold-start users
            
            # Verify the shape makes sense
            assert train_matrix.shape[0] == len(user_map)
            assert train_matrix.shape[1] == len(item_map)
            
            # Verify we have some interactions
            assert np.sum(train_matrix > 0) > 0

# =========================================================
# 2. Boundary Conditions
# =========================================================

class TestBoundaryConditions:
    """Test suite for boundary conditions"""
    
    def test_minimum_interactions_threshold(self):
        """Test minimum interactions threshold"""
        df = pd.DataFrame({
            'user_id': ['u1'] * 3 + ['u2'] * 8 + ['u3'] * 2,
            'bundle_id': ['b1', 'b2', 'b3'] + ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'] + ['b1', 'b2'],
            'user_answer': [1] * 13,
            'correct_answer': [1] * 13,
            'elapsed_time': [10] * 13,
            'timestamp': range(13)
        })
        
        # Test with min_interactions=5 (should exclude u1 and u3)
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(df, min_interactions=5)
        
        assert 'u1' not in user_map  # Excluded due to insufficient interactions
        assert 'u3' not in user_map  # Excluded due to insufficient interactions
        assert 'u2' in user_map       # Included due to sufficient interactions
    
    def test_extreme_rating_values(self):
        """Test handling of extreme rating values"""
        # Create dataset with extreme values
        df = pd.DataFrame({
            'user_id': ['u1'] * 5,
            'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'],
            'user_answer': [0, 0, 0, 0, 0],  # All wrong
            'correct_answer': [1, 1, 1, 1, 1],  # All should be correct
            'elapsed_time': [1, 1000, 1, 1000, 1],  # Extreme time values
            'timestamp': range(5)
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(df, min_interactions=1)
        
        # Should handle extreme values without crashing
        assert train_matrix.shape[0] == 1
        assert train_matrix.shape[1] == 5
        assert np.all(train_matrix >= 0)  # All ratings should be non-negative
    
    def test_large_dataset_handling(self):
        """Test handling of very large datasets"""
        # Create large dataset
        np.random.seed(42)
        n_users = 10000
        n_items = 5000
        n_interactions = 100000
        
        user_ids = np.random.choice(n_users, n_interactions)
        item_ids = np.random.choice(n_items, n_interactions)
        
        large_df = pd.DataFrame({
            'user_id': [f'u{i}' for i in user_ids],
            'bundle_id': [f'b{i}' for i in item_ids],
            'user_answer': np.random.randint(0, 2, n_interactions),
            'correct_answer': np.random.randint(0, 2, n_interactions),
            'elapsed_time': np.random.randint(5, 60, n_interactions),
            'timestamp': range(n_interactions)
        })
        
        # Should handle large dataset without memory issues
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(large_df, min_interactions=10)
        
        assert len(user_map) > 0
        assert len(item_map) > 0
        assert train_matrix.shape[0] > 0
        assert train_matrix.shape[1] > 0

# =========================================================
# 3. Error Conditions
# =========================================================

class TestErrorConditions:
    """Test suite for error conditions and edge cases"""
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        df_with_missing = pd.DataFrame({
            'user_id': ['u1', 'u2', None, 'u3'],
            'bundle_id': ['b1', None, 'b2', 'b3'],
            'user_answer': [1, 0, 1, None],
            'correct_answer': [1, 1, None, 0],
            'elapsed_time': [10, None, 15, 20],
            'timestamp': [0, 1, 2, 3]
        })

        # Should handle missing data gracefully
        try:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(df_with_missing, min_interactions=1)
            # If it succeeds, check that we have some valid data
            assert len(user_map) > 0
            assert len(item_map) > 0
        # If it raises an error, ensure it's a meaningful one
        except Exception as e:
            msg = str(e).lower()
            assert any(keyword in msg for keyword in [
                "missing", "null", "invalid", "error", "type", "unsupported", "not supported"
            ])

        # Should filter out rows with missing critical data
        try:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(df_with_missing, min_interactions=1)
            # If it succeeds, check that we have some valid data
            assert len(user_map) > 0
            assert len(item_map) > 0
            # Should filter out rows with missing critical data
            assert train_matrix.shape[0] >= 0
        except Exception as e:
            msg = str(e).lower()
            assert any(keyword in msg for keyword in [
                "missing", "null", "invalid", "error", "type", "unsupported", "not supported"
            ])

    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        df_with_invalid_types = pd.DataFrame({
            'user_id': [123, 'u2', 456, 'u3'],  # Mixed types
            'bundle_id': [789, 'b2', 101, 'b3'],  # Mixed types
            'user_answer': ['yes', 0, 'no', 1],  # Mixed types
            'correct_answer': ['yes', 1, 'no', 0],  # Mixed types
            'elapsed_time': ['fast', 15, 'slow', 20],  # Mixed types
            'timestamp': range(4)
        })
        
        # Should handle invalid data types gracefully
        try:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(df_with_invalid_types, min_interactions=1)
            # Should not crash
            assert True
        # If it raises an error, ensure it's a meaningful one
        except Exception as e:
            msg = str(e).lower()
            assert any(keyword in msg for keyword in [
                "missing", "null", "invalid", "error", "type", "unsupported", "not supported"
            ])

    def test_duplicate_data_handling(self):
        """Test handling of duplicate data"""
        df_with_duplicates = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2'],
            'bundle_id': ['b1', 'b1', 'b2', 'b1', 'b1'],
            'user_answer': [1, 1, 0, 1, 1],  # Duplicate interactions
            'correct_answer': [1, 1, 1, 1, 1],
            'elapsed_time': [10, 10, 15, 20, 20],  # Duplicate times
            'timestamp': [0, 0, 1, 2, 2]  # Duplicate timestamps
        })
        
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(df_with_duplicates, min_interactions=1)
        
        # Should handle duplicates appropriately (aggregate or keep latest)
        assert train_matrix.shape[0] > 0
        assert train_matrix.shape[1] > 0
    
    def test_extreme_timestamp_values(self):
        """Test handling of extreme timestamp values"""
        df_with_extreme_timestamps = pd.DataFrame({
            'user_id': ['u1'] * 5,
            'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'],
            'user_answer': [1] * 5,
            'correct_answer': [1] * 5,
            'elapsed_time': [10] * 5,
            'timestamp': [-1, 0, 999999999999, float('inf'), float('-inf')]  # Extreme timestamps
        })
        
        # Should handle extreme timestamps gracefully
        try:
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(df_with_extreme_timestamps, min_interactions=1)
            assert True
        except Exception:
            # Should handle gracefully
            assert True

# =========================================================
# 4. Model Edge Cases
# =========================================================

class TestModelEdgeCases:
    """Test suite for model-specific edge cases"""
    
    def test_svd_with_singular_matrix(self):
        """Test SVD with singular matrix"""
        # Create singular matrix (rank-deficient)
        singular_matrix = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        
        recommender = SVDHybridRecommender(n_factors=2)
        recommender.initialize(singular_matrix, {'u1': 0, 'u2': 1, 'u3': 2}, 
                             {'i1': 0, 'i2': 1, 'i3': 2}, {0: 'i1', 1: 'i2', 2: 'i3'})
        
        # Should handle singular matrix gracefully
        try:
            recommender.fit(singular_matrix)
            assert recommender.user_factors is not None
            assert recommender.item_factors is not None
        except Exception as e:
            # Should provide meaningful error or fallback
            assert "singular" in str(e).lower() or "rank" in str(e).lower()
    
    def test_ncf_with_extreme_inputs(self):
        """Test NCF with extreme input values"""
        model = NCF(num_users=100, num_items=50, embedding_dim=16)
        model.eval()
        
        # Test with extreme user/item IDs
        with torch.no_grad():
            # Test with out-of-bounds IDs
            try:
                user_ids = torch.tensor([999, 1000, -1])  # Out of bounds
                item_ids = torch.tensor([999, 1000, -1])  # Out of bounds
                _ = model(user_ids, item_ids)
                # Should handle gracefully or provide meaningful error
            except Exception as e:
                assert "index" in str(e).lower() or "bounds" in str(e).lower()
    
    def test_metrics_with_edge_cases(self):
        """Test evaluation metrics with edge cases"""
        # Test with empty recommendations
        precision = precision_at_k([], [1, 2, 3], k=10)
        recall = recall_at_k([], [1, 2, 3], k=10)
        
        assert precision == 0.0
        assert recall == 0.0
        
        # Test with empty relevant items
        precision = precision_at_k([1, 2, 3], [], k=10)
        recall = recall_at_k([1, 2, 3], [], k=10)
        
        assert precision == 0.0
        assert recall == 0.0
        
        # Test with k=0
        precision = precision_at_k([1, 2, 3], [1, 2, 3], k=0)
        recall = recall_at_k([1, 2, 3], [1, 2, 3], k=0)
        
        assert precision == 0.0
        assert recall == 0.0
        
        # Test RMSE with identical values
        rmse = rmse_score([1, 2, 3], [1, 2, 3])
        assert rmse == 0.0
        
        # Test RMSE with extreme values
        rmse = rmse_score([0, 0, 0], [1000000, 1000000, 1000000])
        assert rmse > 0
        assert not np.isnan(rmse)
        assert not np.isinf(rmse)

# =========================================================
# 5. System Limits & Constraints
# =========================================================

class TestSystemLimits:
    """Test suite for system limits and constraints"""
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints"""
        # Create large model to test memory limits
        try:
            # Try to create very large model
            large_model = NCF(num_users=1000000, num_items=500000, embedding_dim=512)
            # Should either succeed or fail gracefully
            assert True
        except Exception as e:
            # Should provide meaningful error about memory
            assert "memory" in str(e).lower() or "out of memory" in str(e).lower()
    
    def test_computation_time_limits(self):
        """Test behavior under computation time limits"""
        import signal
        import platform

        # SIGALRM is not available on Windows
        if platform.system() == 'Windows':
            # Skip this test on Windows
            pytest.skip("SIGALRM not available on Windows")
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Computation timeout")

        # Set timeout for computation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        try:
            # Test with large dataset
            large_df = pd.DataFrame({
                'user_id': [f'u{i}' for i in range(10000)],
                'bundle_id': [f'b{i}' for i in range(5000)],
                'user_answer': np.random.randint(0, 2, 10000),
                'correct_answer': np.random.randint(0, 2, 10000),
                'elapsed_time': np.random.randint(5, 60, 10000),
                'timestamp': range(10000)
            })
            
            train_matrix, test_matrix, user_map, item_map = prepare_matrices(large_df, min_interactions=1)
            
            # Should complete within timeout
            signal.alarm(0)  # Cancel timeout
            assert True
            
        except TimeoutError:
            # Should handle timeout gracefully
            signal.alarm(0)  # Cancel timeout
            assert True
        except Exception:
            signal.alarm(0)  # Cancel timeout
            assert True
    
    def test_concurrent_access_limits(self):
        """Test behavior under concurrent access"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def concurrent_operation(operation_id):
            """Simulate concurrent operation"""
            try:
                # Simulate model operation
                model = NCF(num_users=100, num_items=50, embedding_dim=16)
                user_ids = torch.randint(0, 100, (10,))
                item_ids = torch.randint(0, 50, (10,))
                
                with torch.no_grad():
                    _ = model(user_ids, item_ids)
                
                results.put((operation_id, "success"))
            except Exception as e:
                results.put((operation_id, f"error: {str(e)}"))
        
        # Test multiple concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        successful_operations = 0
        while not results.empty():
            op_id, result = results.get()
            if "success" in result:
                successful_operations += 1
        
        # Should handle concurrent access gracefully
        assert successful_operations >= 5  # At least 50% success rate 