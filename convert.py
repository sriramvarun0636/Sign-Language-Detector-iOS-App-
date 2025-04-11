import coremltools as ct
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("SignLanguageModel.keras")  # or .h5

# Convert directly from the Keras model (✅ no need for get_concrete_function)
mlmodel = ct.convert(
    model,
    source="tensorflow",
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15
)

# Save the CoreML model
mlmodel.save("SignLanguageModel.mlmodel")
