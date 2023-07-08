import itertools as it


class Spinner:
    def __init__(self):
        self._spinner = it.cycle([char for char in r"/|\\-"])

    def __call__(self, terminate=False):
        char = next(self._spinner)
        print(char, end="\n" if terminate else "\r", flush=True)
