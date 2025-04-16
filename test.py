import os 
import uuid 
static_folder = "static"
target_source_folder = os.path.join(static_folder, "target_source")
os.makedirs(target_source_folder, exist_ok=True)
txt_file_path = os.path.join(target_source_folder, str(uuid.uuid4()) + ".txt")
with open (txt_file_path, "w") as f:
    f.write("gayyyy")
