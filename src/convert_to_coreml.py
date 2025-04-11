import tensorflow as tf
import coremltools as ct
import numpy as np

# Load Keras model
model = tf.keras.models.load_model("sign_language_model.h5")

# Dummy input to trace shape
example_input = np.random.rand(1, 63).astype(np.float32)

# Convert to CoreML
coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(name="dense_input", shape=example_input.shape)],
    convert_to="mlprogram"  # Use "mlprogram" for flexibility (especially on iOS 16+)
)

# Save with correct extension
coreml_model.save("SignLanguageClassifier.mlpackage")