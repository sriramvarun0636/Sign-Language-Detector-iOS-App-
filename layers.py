import coremltools as ct

model = ct.models.MLModel("SignLanguageClassifier.mlmodel")
spec = model.get_spec()

if spec.WhichOneof("Type") == "neuralNetworkClassifier":
    layers = spec.neuralNetworkClassifier.layers
    for i, layer in enumerate(layers):
        print(f"Layer {i + 1}: {layer.name} | Type: {layer.WhichOneof('layer')}")
else:
    print("Not a neural network classifier.")
