import os
import sys


def hard_exit_after_success():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
