"""
Simple script to check if a file exists and print its contents
"""

import os
import sys

def check_file(file_path):
    """
    Check if a file exists and print its contents
    
    Args:
        file_path: Path to the file to check
    """
    print(f"Checking file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return 1
        
    print(f"File exists: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.read()
        print(f"File size: {len(contents)} bytes")
        print(f"First 2000 characters:\n{contents[:2000]}")
        return 0
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        sys.exit(check_file(file_path))
    else:
        print("Usage: python check_file.py <file_path>")
        sys.exit(1) 