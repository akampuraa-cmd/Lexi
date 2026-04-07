"""
main.py — Entry point for Thanos AI.
"""

import os
import sys

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import ThanosApp


def main():
    app = ThanosApp()
    app.mainloop()


if __name__ == "__main__":
    main()
