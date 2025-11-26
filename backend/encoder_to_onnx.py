import tensorflow as tf
import tf2onnx
import onnx

# Path to your TensorFlow encoder
ENCODER_PATH = "models/stacked_encoder.keras"

# Output path for ONNX model
OUTPUT_PATH = "models/stacked_encoder.onnx"

print("🔄 Loading Keras encoder...")
encoder = tf.keras.models.load_model(ENCODER_PATH)

# Dummy input (same dimension as scaler output)
input_shape = encoder.input_shape
dummy_input = tf.random.normal([1, input_shape[1]])

print("🚀 Converting to ONNX format...")
onnx_model, _ = tf2onnx.convert.from_keras(
    encoder,
    input_signature=[tf.TensorSpec(dummy_input.shape, tf.float32)],
    opset=13
)

onnx.save(onnx_model, OUTPUT_PATH)
print(f"✅ Saved ONNX encoder to {OUTPUT_PATH}")
