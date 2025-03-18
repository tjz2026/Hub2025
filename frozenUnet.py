import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.compat.v1.graph_util import convert_variables_to_constants

# 1. Define a simple UNet model
def create_unet_model(input_size=(256, 256, 3)):
    """Create a basic UNet model for demonstration"""
    inputs = Input(input_size)
    
    # Encoder (Downsampling)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Middle
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder (Upsampling)
    up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    concat1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    concat2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 2. Function to convert Keras model to frozen graph
def freeze_keras_model(model, output_path):
    """Convert Keras model to frozen TensorFlow graph"""
    # Convert Keras model to a tf.function
    full_model = tf.function(lambda x: model(x))
    
    # Get concrete function
    input_shape = list(model.inputs[0].shape)
    input_shape[0] = 1  # Set batch size to 1 for inference
    concrete_func = full_model.get_concrete_function(
        tf.TensorSpec(input_shape, model.inputs[0].dtype))
    
    # Get frozen concrete function
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    
    # Save frozen graph
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=os.path.dirname(output_path),
                      name=os.path.basename(output_path),
                      as_text=False)
    
    # Get frozen graph input/output names for later use
    input_names = [input.name.split(':')[0] for input in frozen_func.inputs 
                  if input.dtype != tf.resource]
    output_names = [output.name.split(':')[0] for output in frozen_func.outputs]
    
    print(f"Model frozen successfully at: {output_path}")
    print(f"Frozen model inputs: {input_names}")
    print(f"Frozen model outputs: {output_names}")
    
    return input_names, output_names

# 3. Function to optimize the frozen graph
def optimize_frozen_graph(input_graph_path, output_graph_path, input_names, output_names):
    """Optimize the frozen graph for inference"""
    # Read the frozen graph
    input_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(input_graph_path, "rb") as f:
        input_graph_def.ParseFromString(f.read())
    
    # Optimize for inference
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names,
        output_names,
        tf.float32.as_datatype_enum)
    
    # Save the optimized graph
    with tf.io.gfile.GFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    
    print(f"Graph optimized successfully at: {output_graph_path}")
    return output_graph_path

# 4. Function to run inference with the frozen graph
def run_inference_frozen_graph(graph_path, input_data, input_node_name, output_node_name):
    """Run inference using the frozen graph"""
    # Load frozen graph
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Create graph
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
        # Create session with graph optimization config
        config = tf.compat.v1.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            # Get input and output tensors
            input_tensor = graph.get_tensor_by_name(f"{input_node_name}:0")
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
            
            # Run inference
            start_time = time.time()
            result = sess.run(output_tensor, {input_tensor: input_data})
            inference_time = time.time() - start_time
            
    return result, inference_time

# 5. Function to run inference using the original Keras model (for comparison)
def run_inference_keras(model, input_data):
    """Run inference using the original Keras model"""
    start_time = time.time()
    result = model.predict(input_data)
    inference_time = time.time() - start_time
    
    return result, inference_time

# 6. Complete pipeline
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    output_dir = "./unet_exported_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create and save UNet model
    print("Creating UNet model...")
    input_shape = (256, 256, 3)
    unet_model = create_unet_model(input_shape)
    unet_model.summary()
    
    # Step 2: Save Keras model (optional)
    keras_model_path = os.path.join(output_dir, "unet_keras_model.h5")
    unet_model.save(keras_model_path)
    print(f"Keras model saved at: {keras_model_path}")
    
    # Step 3: Freeze the model
    frozen_graph_path = os.path.join(output_dir, "frozen_unet.pb")
    input_names, output_names = freeze_keras_model(unet_model, frozen_graph_path)
    
    # Step 4: Optimize the frozen graph
    optimized_graph_path = os.path.join(output_dir, "optimized_unet.pb")
    optimize_frozen_graph(frozen_graph_path, optimized_graph_path, input_names, output_names)
    
    # Step 5: Create test data
    print("\nPreparing test data for inference...")
    test_image = np.random.random((1, 256, 256, 3)).astype(np.float32)
    
    # Step 6: Run inference with Keras model
    print("\nRunning inference with Keras model...")
    keras_result, keras_time = run_inference_keras(unet_model, test_image)
    print(f"Keras model inference time: {keras_time:.4f} seconds")
    
    # Step 7: Run inference with frozen graph
    print("\nRunning inference with frozen graph...")
    frozen_result, frozen_time = run_inference_frozen_graph(
        frozen_graph_path, test_image, input_names[0], output_names[0])
    print(f"Frozen graph inference time: {frozen_time:.4f} seconds")
    
    # Step 8: Run inference with optimized graph
    print("\nRunning inference with optimized graph...")
    optimized_result, optimized_time = run_inference_frozen_graph(
        optimized_graph_path, test_image, input_names[0], output_names[0])
    print(f"Optimized graph inference time: {optimized_time:.4f} seconds")
    
    # Step 9: Compare results and speedup
    print("\nResults comparison:")
    print(f"Keras output shape: {keras_result.shape}")
    print(f"Frozen graph output shape: {frozen_result.shape}")
    print(f"Optimized graph output shape: {optimized_result.shape}")
    
    print("\nPerformance comparison:")
    print(f"Keras model inference time: {keras_time:.4f} seconds (baseline)")
    print(f"Frozen graph speedup: {keras_time/frozen_time:.2f}x")
    print(f"Optimized graph speedup: {keras_time/optimized_time:.2f}x")
    
    # Step 10: Verify output consistency
    keras_mean = np.mean(keras_result)
    frozen_mean = np.mean(frozen_result)
    optimized_mean = np.mean(optimized_result)
    
    print("\nOutput consistency check:")
    print(f"Keras output mean: {keras_mean:.6f}")
    print(f"Frozen graph output mean: {frozen_mean:.6f}")
    print(f"Optimized graph output mean: {optimized_mean:.6f}")
    
    # Calculate absolute difference
    frozen_diff = np.abs(keras_result - frozen_result).mean()
    optimized_diff = np.abs(keras_result - optimized_result).mean()
    
    print(f"Mean absolute difference (Keras vs Frozen): {frozen_diff:.6f}")
    print(f"Mean absolute difference (Keras vs Optimized): {optimized_diff:.6f}")

if __name__ == "__main__":
    main()