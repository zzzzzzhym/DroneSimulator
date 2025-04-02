import pickle
import os

class RobustPickle:
    """Regular pickle requires the class to be defined in the same file as the object being pickled.
    This class micmics the pytorch model saving and loading process. Only the __dict__ is saved and loaded.
    This allows the class to be defined in a different file.
    """
    @staticmethod
    def save(obj, file_path):
        """Save the object to a file using pickle"""
        with open(file_path, 'wb') as file:
            pickle.dump(obj.__dict__, file)
        print(f"Object saved to {os.path.relpath(file_path)}")

    def load(obj, file_path, strict=True):
        with open(file_path, 'rb') as f:
            state_dict = pickle.load(f)
        missing_keys = []
        unexpected_keys = []

        for key, value in state_dict.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
            else:
                unexpected_keys.append(key)

        if strict:
            # Use obj.state_dict() if defined, otherwise fallback to __dict__
            current_state = (obj.__dict__)
            current_keys = set(current_state)
            loaded_keys = set(state_dict)

            for key in current_keys - loaded_keys:
                missing_keys.append(key)

            if missing_keys or unexpected_keys:
                raise ValueError(
                    f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
                )