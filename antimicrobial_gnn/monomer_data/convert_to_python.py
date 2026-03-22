import pickle


def convert_numpy_to_python(obj):
    """Recursively converts numpy types to Python native types"""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.number):
        return obj.item()  # Convert numpy numbers to Python numbers
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")


# Load the original file
input_file = "monomer_data/rdkit_monomer_features.pkl"
output_file = "monomer_data/rdkit_monomer_features_2.py"

try:
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Convert numpy types to Python types
    converted_data = convert_numpy_to_python(data)

    # Save back to a new file
    with open(output_file, "wb") as f:
        pickle.dump(converted_data, f)

    print(f"Successfully converted and saved to {output_file}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
