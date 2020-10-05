#!/usr/bin/env python
import unittest
import os

def load_tests(loader, tests, pattern):
    suite = unittest.defaultTestLoader.discover(os.path.join(os.path.dirname(__file__),'genicam_tests'), pattern='*test.py')
    return suite

if __name__ == "__main__":
    unittest.main()