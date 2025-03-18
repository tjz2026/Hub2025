import tensorflow as tf
import numpy as np
import threading
import time

# -----------------------------------------------------------------------------
# Dummy Model Setup
# -----------------------------------------------------------------------------
# Create a simple model. In a real scenario, replace this with your actual model.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile()  # Inference does not require compilation, but we call it here for completeness.

# -----------------------------------------------------------------------------
# Dummy Dataset Generator
# -----------------------------------------------------------------------------
def generator():
    req_id = 0
    while True:
        # Each request provides a variable number of images; here we use a fixed number for simplicity.
        images = np.random.rand(4, 256, 256, 3).astype(np.float32)
        yield req_id, images
        req_id += 1

# Create a tf.data.Dataset from the generator.
dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),               # request id
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32)  # images with variable batch dimension
    )
)

# Prefetch one element to decouple data loading.
dataset = dataset.prefetch(1)

# -----------------------------------------------------------------------------
# Asynchronous Copy and Save Function
# -----------------------------------------------------------------------------
def save_result(request_id, result):
    # Here, you could compress the result or write it to a memory drive.
    print(f"Saving result for request {request_id}, result shape: {result.shape}")

def async_copy_and_save(request_id, predictions):
    start_copy = time.time()
    print(f"[Async Copy] Request {request_id}: Copy started at {start_copy:.4f}")
    # Simulate a slow GPU-to-CPU copy by adding an artificial delay.
    time.sleep(1.0)
    result = predictions.numpy()  # This triggers the actual GPU-to-CPU copy.
    end_copy = time.time()
    print(f"[Async Copy] Request {request_id}: Copy finished at {end_copy:.4f} (duration {(end_copy - start_copy):.4f}s)")
    save_result(request_id, result)

# -----------------------------------------------------------------------------
# Inference Worker with Logging to Demonstrate Overlap
# -----------------------------------------------------------------------------
def inference_worker(dataset):
    for req_id, images in dataset:
        start_infer = time.time()
        print(f"[Inference] Request {req_id.numpy()}: Inference started at {start_infer:.4f}")
        predictions = model(images)  # Run inference on GPU.
        infer_end = time.time()
        print(f"[Inference] Request {req_id.numpy()}: Inference finished at {infer_end:.4f} (duration {(infer_end - start_infer):.4f}s)")
        
        # Start asynchronous GPU-to-CPU copy in a background thread.
        threading.Thread(
            target=async_copy_and_save,
            args=(req_id.numpy(), predictions),
            daemon=True
        ).start()
        
        # Optionally simulate a short delay before processing the next request.
        time.sleep(0.5)

# -----------------------------------------------------------------------------
# Run the Inference Worker
# -----------------------------------------------------------------------------
inference_worker(dataset)
