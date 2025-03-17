import os

def clean_init_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "__init__.py":
                file_path = os.path.join(root, file)
                print(f"Cleaning {file_path}")
                
                # Create a clean file
                with open(file_path, 'w', encoding='utf-8') as f:
                    if "core" in file_path:
                        f.write('"""Core functionality for GraphRAG"""\n')
                    elif "connectors" in file_path:
                        f.write('"""Database connectors for GraphRAG"""\n')
                    elif "cli" in file_path:
                        f.write('"""Command-line interface for GraphRAG"""\n')
                    elif "data" in file_path:
                        f.write('"""Data handling utilities for GraphRAG"""\n')
                    elif "models" in file_path:
                        f.write('"""Model definitions for GraphRAG"""\n')
                    elif "tests/integration" in file_path:
                        f.write('"""Integration tests for GraphRAG"""\n')
                    elif "tests/unit" in file_path:
                        f.write('"""Unit tests for GraphRAG"""\n')
                    elif "tests" in file_path:
                        f.write('"""Tests for GraphRAG"""\n')
                    elif "utils" in file_path:
                        f.write('"""Utility functions for GraphRAG"""\n')
                    else:
                        f.write('"""GraphRAG: Graph-based Retrieval Augmented Generation"""\n\n__version__ = "0.1.0"\n')

if __name__ == "__main__":
    clean_init_files("graphrag") 