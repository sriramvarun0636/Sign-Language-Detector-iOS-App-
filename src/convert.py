import coremltools as ct
import tensorflow as tf

# Load your .h5 Keras model
model = tf.keras.models.load_model("sign_language_model.h5")

# Convert the model with correct input name
mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(name="dense_input", shape=(1, 63))],  # 👈 your model expects this
)

# Save the converted model
mlmodel.save("SignLanguageClassifier.mlpackage")