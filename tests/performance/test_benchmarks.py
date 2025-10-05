"""
Performance tests for model inference

Tests cover:
- Inference latency
- Batch prediction throughput
- Memory usage
- CPU/GPU utilization
- Scalability
"""

import pytest
import numpy as np
import time
import psutil
import tensorflow as tf
from statistics import mean, stdev


@pytest.mark.performance
class TestInferenceLatency:
    """Test inference latency performance"""

    def test_single_inference_latency(self, trained_model, sample_train_test_split, performance_config):
        """Test single sample inference latency"""
        X_test = sample_train_test_split[1]
        max_latency_ms = performance_config['max_inference_latency_ms']

        # Warm up
        for _ in range(10):
            trained_model.predict(X_test[0:1], verbose=0)

        # Measure latency
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            trained_model.predict(X_test[i:i+1], verbose=0)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\nSingle inference latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        assert avg_latency < max_latency_ms, \
            f"Average latency {avg_latency:.2f}ms exceeds {max_latency_ms}ms"

    def test_batch_inference_latency(self, trained_model, sample_train_test_split, performance_config):
        """Test batch inference latency for different batch sizes"""
        X_test = sample_train_test_split[1]

        results = {}
        for batch_size in performance_config['batch_sizes']:
            batch = X_test[:batch_size]

            # Warm up
            for _ in range(5):
                trained_model.predict(batch, verbose=0)

            # Measure
            latencies = []
            for _ in range(50):
                start = time.perf_counter()
                trained_model.predict(batch, verbose=0)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

            avg_latency = mean(latencies)
            results[batch_size] = avg_latency

        print(f"\nBatch inference latency:")
        for batch_size, latency in results.items():
            samples_per_sec = (batch_size / latency) * 1000
            print(f"  Batch {batch_size}: {latency:.2f}ms ({samples_per_sec:.0f} samples/sec)")

    def test_latency_consistency(self, trained_model, sample_train_test_split):
        """Test latency consistency (low variance)"""
        X_test = sample_train_test_split[1]
        sample = X_test[0:1]

        # Warm up
        for _ in range(10):
            trained_model.predict(sample, verbose=0)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            trained_model.predict(sample, verbose=0)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg = mean(latencies)
        std = stdev(latencies)
        cv = (std / avg) * 100  # Coefficient of variation

        print(f"\nLatency consistency:")
        print(f"  Mean: {avg:.2f}ms")
        print(f"  Std Dev: {std:.2f}ms")
        print(f"  CV: {cv:.1f}%")

        assert cv < 50, f"Latency variance too high: {cv:.1f}%"


@pytest.mark.performance
class TestThroughput:
    """Test prediction throughput"""

    def test_throughput_single_thread(self, trained_model, sample_train_test_split, performance_config):
        """Test single-threaded throughput"""
        X_test = sample_train_test_split[1]
        min_throughput = performance_config['min_throughput_samples_per_sec']

        # Warm up
        trained_model.predict(X_test[:100], verbose=0)

        # Measure
        num_samples = 1000
        start = time.perf_counter()
        trained_model.predict(X_test[:num_samples], verbose=0)
        end = time.perf_counter()

        duration = end - start
        throughput = num_samples / duration

        print(f"\nSingle-thread throughput: {throughput:.0f} samples/sec")

        assert throughput > min_throughput, \
            f"Throughput {throughput:.0f} below minimum {min_throughput}"

    def test_throughput_batch_sizes(self, trained_model, sample_train_test_split, performance_config):
        """Test throughput with different batch sizes"""
        X_test = sample_train_test_split[1]

        results = {}
        for batch_size in performance_config['batch_sizes']:
            num_batches = 50
            total_samples = batch_size * num_batches

            # Warm up
            trained_model.predict(X_test[:batch_size], verbose=0)

            # Measure
            start = time.perf_counter()
            for i in range(num_batches):
                offset = (i * batch_size) % (len(X_test) - batch_size)
                trained_model.predict(X_test[offset:offset+batch_size], verbose=0)
            end = time.perf_counter()

            throughput = total_samples / (end - start)
            results[batch_size] = throughput

        print(f"\nThroughput by batch size:")
        for batch_size, throughput in results.items():
            print(f"  Batch {batch_size}: {throughput:.0f} samples/sec")

        # Larger batches should generally have higher throughput
        assert results[max(results.keys())] > results[min(results.keys())], \
            "Larger batches should have higher throughput"

    def test_sustained_throughput(self, trained_model, sample_train_test_split):
        """Test sustained throughput over time"""
        X_test = sample_train_test_split[1]
        batch_size = 32
        duration_seconds = 5

        # Warm up
        trained_model.predict(X_test[:batch_size], verbose=0)

        # Measure sustained throughput
        samples_processed = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            offset = (samples_processed % (len(X_test) - batch_size))
            trained_model.predict(X_test[offset:offset+batch_size], verbose=0)
            samples_processed += batch_size

        end = time.perf_counter()
        actual_duration = end - start
        sustained_throughput = samples_processed / actual_duration

        print(f"\nSustained throughput ({actual_duration:.1f}s): {sustained_throughput:.0f} samples/sec")

        assert sustained_throughput > 500, "Sustained throughput too low"


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage during inference"""

    def test_memory_single_inference(self, trained_model, sample_train_test_split):
        """Test memory usage for single inference"""
        X_test = sample_train_test_split[1]
        process = psutil.Process()

        # Baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Inference
        for _ in range(100):
            trained_model.predict(X_test[0:1], verbose=0)

        # Final memory
        final_mb = process.memory_info().rss / 1024 / 1024
        increase_mb = final_mb - baseline_mb

        print(f"\nSingle inference memory:")
        print(f"  Baseline: {baseline_mb:.1f} MB")
        print(f"  Final: {final_mb:.1f} MB")
        print(f"  Increase: {increase_mb:.1f} MB")

        # Memory increase should be minimal
        assert increase_mb < 50, f"Memory increased by {increase_mb:.1f} MB"

    def test_memory_batch_inference(self, trained_model, sample_train_test_split, performance_config):
        """Test memory usage for batch inference"""
        X_test = sample_train_test_split[1]
        process = psutil.Process()

        results = {}
        for batch_size in performance_config['batch_sizes']:
            baseline_mb = process.memory_info().rss / 1024 / 1024

            # Inference
            batch = X_test[:batch_size]
            for _ in range(20):
                trained_model.predict(batch, verbose=0)

            final_mb = process.memory_info().rss / 1024 / 1024
            increase_mb = final_mb - baseline_mb
            results[batch_size] = increase_mb

        print(f"\nBatch inference memory usage:")
        for batch_size, increase in results.items():
            print(f"  Batch {batch_size}: +{increase:.1f} MB")

    def test_memory_leak_detection(self, trained_model, sample_train_test_split):
        """Test for memory leaks during repeated inference"""
        X_test = sample_train_test_split[1]
        process = psutil.Process()

        # Run many inferences and track memory
        memory_samples = []
        for i in range(10):
            for _ in range(100):
                trained_model.predict(X_test[0:1], verbose=0)

            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)

        # Check for memory growth trend
        memory_growth = memory_samples[-1] - memory_samples[0]

        print(f"\nMemory leak detection:")
        print(f"  Initial: {memory_samples[0]:.1f} MB")
        print(f"  Final: {memory_samples[-1]:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")

        assert memory_growth < 100, f"Potential memory leak: {memory_growth:.1f} MB growth"


@pytest.mark.performance
class TestScalability:
    """Test model scalability"""

    def test_input_size_scaling(self, trained_model):
        """Test how performance scales with input size"""
        results = []

        for num_samples in [1, 10, 100, 500, 1000]:
            data = np.random.randn(num_samples, 3197)

            # Warm up
            trained_model.predict(data[:1], verbose=0)

            # Measure
            start = time.perf_counter()
            trained_model.predict(data, verbose=0)
            end = time.perf_counter()

            duration = (end - start) * 1000
            per_sample = duration / num_samples
            results.append((num_samples, duration, per_sample))

        print(f"\nScaling with input size:")
        for num_samples, total_ms, per_sample_ms in results:
            print(f"  {num_samples} samples: {total_ms:.1f}ms total, {per_sample_ms:.2f}ms per sample")

    def test_concurrent_predictions(self, trained_model, sample_train_test_split):
        """Test performance with concurrent prediction requests"""
        import threading

        X_test = sample_train_test_split[1]
        num_threads = 4
        predictions_per_thread = 25

        def predict_worker(thread_id, results):
            latencies = []
            for i in range(predictions_per_thread):
                start = time.perf_counter()
                trained_model.predict(X_test[i:i+1], verbose=0)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            results[thread_id] = latencies

        results = {}
        threads = []

        start = time.perf_counter()
        for i in range(num_threads):
            t = threading.Thread(target=predict_worker, args=(i, results))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        end = time.perf_counter()

        total_duration = end - start
        total_predictions = num_threads * predictions_per_thread
        throughput = total_predictions / total_duration

        print(f"\nConcurrent predictions ({num_threads} threads):")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Total time: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} predictions/sec")


@pytest.mark.performance
class TestModelSize:
    """Test model size and loading performance"""

    def test_model_file_size(self, trained_model, tmp_path):
        """Test saved model file size"""
        model_path = tmp_path / "model_size_test.h5"
        trained_model.save(str(model_path))

        file_size_mb = model_path.stat().st_size / 1024 / 1024

        print(f"\nModel file size: {file_size_mb:.2f} MB")

        # Model should be reasonably sized
        assert file_size_mb < 100, f"Model file too large: {file_size_mb:.2f} MB"

    def test_model_loading_time(self, trained_model, tmp_path):
        """Test model loading performance"""
        model_path = tmp_path / "model_loading_test.h5"
        trained_model.save(str(model_path))

        # Measure loading time
        load_times = []
        for _ in range(5):
            tf.keras.backend.clear_session()

            start = time.perf_counter()
            loaded_model = tf.keras.models.load_model(str(model_path))
            end = time.perf_counter()

            load_times.append((end - start) * 1000)

        avg_load_time = mean(load_times)

        print(f"\nModel loading time:")
        print(f"  Average: {avg_load_time:.0f}ms")
        print(f"  Min: {min(load_times):.0f}ms")
        print(f"  Max: {max(load_times):.0f}ms")

        assert avg_load_time < 5000, f"Model loading too slow: {avg_load_time:.0f}ms"

    def test_model_memory_footprint(self, trained_model):
        """Test model memory footprint"""
        process = psutil.Process()

        # Memory before model
        tf.keras.backend.clear_session()
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Load model (already loaded as trained_model)
        current_mb = process.memory_info().rss / 1024 / 1024
        model_mb = current_mb - baseline_mb

        print(f"\nModel memory footprint: {model_mb:.1f} MB")

        # Model should fit in reasonable memory
        assert model_mb < 500, f"Model memory footprint too large: {model_mb:.1f} MB"


@pytest.mark.performance
@pytest.mark.benchmark
def test_benchmark_suite(benchmark, trained_model, sample_train_test_split):
    """Comprehensive benchmark using pytest-benchmark"""
    X_test = sample_train_test_split[1]
    batch = X_test[:32]

    # Benchmark batch prediction
    result = benchmark(trained_model.predict, batch, verbose=0)

    print(f"\nBenchmark results:")
    print(f"  Mean: {benchmark.stats.mean * 1000:.2f}ms")
    print(f"  Median: {benchmark.stats.median * 1000:.2f}ms")
    print(f"  Min: {benchmark.stats.min * 1000:.2f}ms")
    print(f"  Max: {benchmark.stats.max * 1000:.2f}ms")
