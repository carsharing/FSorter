"""Entry script for the packaged Windows app."""

import multiprocessing
import sys

from filesorter.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
