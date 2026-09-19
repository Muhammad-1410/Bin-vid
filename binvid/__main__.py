from __future__ import annotations
import multiprocessing as mp
import sys

def run() -> None:
    mp.freeze_support()
    if len(sys.argv) > 1:
        from binvid.cli import main as cli_main
        cli_main()
    else:
        from binvid.app import main as app_main
        app_main()
if __name__ == '__main__':
    run()
