import logging
import time
import csv
import torch

from logging_setup import setup_logging
from neural.chess_model_hybrid import ThreePlayerHybrid
from neural.model_factory import model_factory


class InferenceSpeedResults:
    def __init__(
            self,
            batch_size: int,
            total_inferences: int,
            avg_time_per_inference: float,
            avg_inferences_per_second: float,
    ):
        self.batch_size = batch_size
        self.total_inferences = total_inferences
        self.avg_time_per_inference = avg_time_per_inference
        self.avg_inferences_per_second = avg_inferences_per_second

    def __str__(self):
        return (
            f"Batch size: {self.batch_size}. Total inferences: {self.total_inferences}. "
            f"Average time per inference: {self.avg_time_per_inference:.6f} s. "
            f"Average inferences per second: {self.avg_inferences_per_second:.2f}"
        )


def run_stress_test(
        model: ThreePlayerHybrid, batch_size: int, num_inferences: int
) -> InferenceSpeedResults:
    """
    Runs a stress test on the given ChessModelHybrid to measure inference speed.
    """
    logging.info(
        f"Starting stress test for batch size {batch_size} with {num_inferences} inferences."
    )

    # Prepare dummy input
    dummy_input = torch.randn(
        batch_size, model.seq_len, model.input_features, device=model.device
    )

    # Pre-warmup runs
    logging.info("Performing warmup runs...")
    for _ in range(10):  # A few warmup runs to stabilize performance
        with torch.no_grad():
            model(dummy_input)
    logging.info("Warmup complete.")

    # Actual measurement
    start_time = time.perf_counter()
    for _ in range(num_inferences // batch_size):
        with torch.no_grad():
            model(dummy_input)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_inferences_made = (num_inferences // batch_size) * batch_size
    avg_time_per_inference = total_time / total_inferences_made
    avg_inferences_per_second = total_inferences_made / total_time if total_time > 0 else float("inf")

    results = InferenceSpeedResults(
        batch_size=batch_size,
        total_inferences=total_inferences_made,
        avg_time_per_inference=avg_time_per_inference,
        avg_inferences_per_second=avg_inferences_per_second,
    )
    return results


if __name__ == "__main__":
    setup_logging()

    # Model configuration (should match what's used for training/inference)
    model = model_factory("ChessDomain")

    if not isinstance(model, ThreePlayerHybrid):
        raise TypeError("Loaded model is not an instance of ThreePlayerHybrid.")

    model.eval()  # Set to evaluation mode
    model.precompute_for_inference()  # Precompute buffers for faster inference

    # Move model to GPU if available
    device_name = "cpu"
    if torch.cuda.is_available():
        model.cuda()
        device_name = "cuda"
        logging.info("Model moved to CUDA.")
    else:
        logging.info("CUDA not available, running on CPU.")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    total_inferences_per_batch_size = 100

    results_csv = "stress_test_results.csv"
    fieldnames = [
        "timestamp",
        "batch_size",
        "total_inferences",
        "avg_time_per_inference",
        "avg_inferences_per_second",
        "device",
    ]

    logging.info("--- Starting ChessModelHybrid Stress Test ---")
    with open(results_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for bs in batch_sizes:
            net = model
            if torch.cuda.is_available():
                # compile for potential speed; keep using `model` for attribute access in the test
                net = torch.compile(model, mode="default", fullgraph=True)

            results = run_stress_test(model, bs, total_inferences_per_batch_size)

            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "batch_size": results.batch_size,
                "total_inferences": results.total_inferences,
                "avg_time_per_inference": f"{results.avg_time_per_inference:.6f}",
                "avg_inferences_per_second": f"{results.avg_inferences_per_second:.2f}",
                "device": device_name,
            })
            csvfile.flush()

            logging.info(f"Results: {results}")

    logging.info(f"--- Stress Test Complete --- Results written to {results_csv}")