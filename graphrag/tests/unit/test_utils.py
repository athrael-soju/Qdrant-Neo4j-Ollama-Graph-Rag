"""
Unit tests for utility functions.
"""

import unittest
import os
import tempfile
from graphrag.utils.common import get_file_extension, ensure_directory

class TestCommonUtils(unittest.TestCase):
    """Test common utility functions."""
    
    def test_get_file_extension(self):
        """Test get_file_extension function."""
        self.assertEqual(get_file_extension("file.txt"), "txt")
        self.assertEqual(get_file_extension("file.TXT"), "txt")
        self.assertEqual(get_file_extension("file"), "")
        self.assertEqual(get_file_extension("file.tar.gz"), "gz")
        self.assertEqual(get_file_extension("/path/to/file.pdf"), "pdf")
        
    def test_ensure_directory(self):
        """Test ensure_directory function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_dir")
            self.assertFalse(os.path.exists(test_dir))
            
            ensure_directory(test_dir)
            self.assertTrue(os.path.exists(test_dir))
            
            # Should not raise an error if directory already exists
            ensure_directory(test_dir)
            
if __name__ == "__main__":
    unittest.main() 