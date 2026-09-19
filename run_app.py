#!/usr/bin/env python3
"""Launch script for the binvid Gradio web application."""

import multiprocessing as mp
from binvid.app import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
