

file_path = './data/target_source.txt'
import os 
import shutil



def remove_all_files_except(folder_path, file_to_keep):
    """
    Removes all files in the specified folder except for the file_to_keep.
    
    Parameters:
        folder_path (str): The path to the folder.
        file_to_keep (str): The name of the file to keep.
    """
    # Iterate over each entry in the folder
    for filename in os.listdir(folder_path):
        # Skip the file we want to keep
        if filename == file_to_keep:
            continue
        
        # Construct the full path of the file
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (skip directories)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        else:
            print(f"Skipping non-file: {file_path}")

# Example usage:
if __name__ == "__main__":
    file_path = "./data/target_source.txt"
    folder = './data'
    file_to_keep = "target_source.txt"

    # Check if the path exists (file or directory)
    if os.path.exists(file_path):
        print("The path exists.")
    else:
        raise FileNotFoundError
    remove_all_files_except(folder, file_to_keep)
    os.makedirs("output_train_log", exist_ok=True)
    os.makedirs("output_tensorboard_log", exist_ok=True)

