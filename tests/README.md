# Testing Suite Documentation

This document provides comprehensive information about the testing strategy and test suite for the Data-Driven Personalized Educational Content Recommendation System.

## Testing Strategy

Our testing approach follows a **multi-layered testing strategy** to ensure comprehensive coverage and quality assurance:

### 1. Unit Tests (`test_*.py`)
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: Core algorithms, utility functions, data processing
- **Tools**: pytest, unittest.mock
- **Files**: `test_metrics.py`, `test_preprocessing.py`, `test_schemas.py`, `test_mappings.py`

### 2. ML Model Tests (`test_ml_models.py`)
- **Purpose**: Test machine learning models and algorithms
- **Coverage**: SVD recommender, NCF model, content similarity
- **Features**: Model training, prediction, persistence, edge cases
- **Validation**: Model outputs, convergence, error handling

### 3. Integration Tests (`test_integration.py`)
- **Purpose**: Test complete workflows and system integration
- **Coverage**: Full recommendation pipeline, API endpoints, dashboard
- **Features**: End-to-end testing, API validation, data flow
- **Tools**: FastAPI TestClient, mocked components

### 4. Performance Tests (`test_performance.py`)
- **Purpose**: Benchmark system performance and scalability
- **Coverage**: Response times, memory usage, concurrent operations
- **Features**: Load testing, resource monitoring, scalability analysis
- **Tools**: psutil, threading, time measurement

### 5. Edge Case Tests (`test_edge_cases.py`)
- **Purpose**: Test extreme scenarios and error conditions
- **Coverage**: Empty data, invalid inputs, system limits
- **Features**: Error handling, boundary conditions, stress testing
- **Validation**: Graceful degradation, meaningful error messages

### 6. Coverage Tests (`test_coverage.py`)
- **Purpose**: Ensure comprehensive code coverage
- **Coverage**: All source files, critical paths, error handling
- **Features**: Coverage reporting, threshold enforcement
- **Tools**: coverage.py, HTML reports

## Test Categories

### Core Algorithm Tests
```python
# SVD Hybrid Recommender
- Initialization and configuration
- Model training and fitting
- Rating prediction
- Recommendation generation
- Error handling and validation

# NCF Model
- Model architecture
- Forward pass computation
- Batch processing
- Memory efficiency
- Model persistence
```

### API Endpoint Tests
```python
# FastAPI Endpoints
- Health check endpoint
- User management endpoints
- Recommendation endpoints
- Error handling and validation
- Response format validation
```

### Performance Benchmarks
```python
# Performance Metrics
- Training time benchmarks
- Inference time benchmarks
- Memory usage monitoring
- Scalability testing
- Concurrent operation testing
```

### Edge Cases and Error Handling
```python
# Extreme Scenarios
- Empty datasets
- Single user/item datasets
- Sparse/dense datasets
- Invalid data types
- Missing data handling
- System resource limits
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_metrics.py

# Run specific test function
pytest tests/test_metrics.py::test_precision_at_k
```

### Test Categories
```bash
# Run only unit tests
pytest -m "not integration and not performance"

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only edge case tests
pytest -m edge_cases

# Run only coverage tests
pytest -m coverage
```

### Coverage Analysis
```bash
# Run with coverage reporting
pytest --cov=src --cov-report=html

# Generate coverage report
pytest --cov=src --cov-report=term-missing

# Check coverage threshold
pytest --cov-fail-under=80
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/test_performance.py -v

# Run with timing information
pytest tests/test_performance.py --durations=10
```

## Test Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=80
```

### Coverage Configuration
```python
# Coverage settings in test_coverage.py
cov = coverage.Coverage(
    source=['src'],
    omit=[
        '*/tests/*',
        '*/__pycache__/*',
        '*/venv/*',
        '*/env/*',
        '*/\.venv/*',
        '*/\.env/*'
    ]
)
```

## Test Data Management

### Sample Data Generation
```python
@pytest.fixture
def sample_data(self):
    """Create comprehensive sample dataset"""
    return pd.DataFrame({
        'user_id': ['u1'] * 10 + ['u2'] * 8 + ['u3'] * 12,
        'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'] * 6,
        'user_answer': [1, 0, 1, 1, 0] * 6,
        'correct_answer': [1, 1, 0, 1, 1] * 6,
        'elapsed_time': [10, 15, 8, 12, 20] * 6,
        'timestamp': range(30)
    })
```

### Mock Data for Testing
```python
# Mock API responses
mock_users_response = {"users": ["u1", "u2", "u3"]}
mock_history_response = {
    "history": [
        {
            "question_id": "q1",
            "bundle_id": "b1",
            "timestamp": "2024-01-01T12:00:00",
            "is_correct": True,
            "elapsed_time": 10.5,
            "part": "Part 1",
            "subjects": ["listening"]
        }
    ],
    "total_interactions": 1
}
```

## Quality Assurance

### Test Quality Metrics
- **Coverage Threshold**: 80% minimum code coverage
- **Performance Benchmarks**: Response time < 100ms per recommendation
- **Memory Efficiency**: < 500MB memory increase for large datasets
- **Error Handling**: Graceful degradation for all error conditions

### Continuous Integration
```yaml
# Example CI configuration
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
    pytest --cov=src --cov-report=html

- name: Check coverage threshold
  run: |
    pytest --cov=src --cov-fail-under=80
```

### Test Maintenance
- **Regular Updates**: Tests updated with code changes
- **Performance Monitoring**: Continuous performance benchmarking
- **Coverage Tracking**: Automated coverage reporting
- **Documentation**: Comprehensive test documentation

## Best Practices

### Writing Effective Tests
1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **Comprehensive Coverage**: Test both success and failure cases
4. **Performance Awareness**: Monitor test execution time
5. **Maintainability**: Keep tests simple and readable

### Test Organization
1. **Logical Grouping**: Group related tests in classes
2. **Fixture Reuse**: Use fixtures for common setup
3. **Mock Appropriately**: Mock external dependencies
4. **Edge Case Coverage**: Test boundary conditions
5. **Error Scenarios**: Test error handling paths

### Performance Testing Guidelines
1. **Baseline Establishment**: Establish performance baselines
2. **Regression Detection**: Monitor for performance regressions
3. **Resource Monitoring**: Track memory and CPU usage
4. **Scalability Testing**: Test with varying dataset sizes
5. **Concurrent Testing**: Test system under load

## Troubleshooting

### Common Issues
1. **Import Errors**: Check Python path configuration
2. **Mock Issues**: Verify mock setup and teardown
3. **Performance Failures**: Check system resources
4. **Coverage Gaps**: Review uncovered code paths
5. **Test Dependencies**: Ensure proper test isolation

### Debugging Tests
```bash
# Run with debug output
pytest -v -s

# Run specific failing test
pytest tests/test_specific.py::test_function -v -s

# Check test discovery
pytest --collect-only
```

This comprehensive testing suite ensures the reliability, performance, and maintainability of the recommendation system while providing confidence in the quality of the implementation. 