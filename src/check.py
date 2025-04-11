import pickle

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

print("Label Map:", label_map)