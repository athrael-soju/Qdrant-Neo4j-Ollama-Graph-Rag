import os

def check_file_for_null_bytes(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            null_positions = [i for i, byte in enumerate(content) if byte == 0]
            if null_positions:
                print(f"Found {len(null_positions)} null bytes in {file_path} at positions: {null_positions[:10]}...")
                return True
            else:
                return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def scan_directory(directory):
    found_null = False
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if check_file_for_null_bytes(file_path):
                    found_null = True
    
    if not found_null:
        print("No null bytes found in any Python files.")

if __name__ == "__main__":
    scan_directory('graphrag') 