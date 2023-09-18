import numpy as np
import sys


def main():
    print(f"python is: {sys.executable}")

    arr = np.arange(10_000, dtype=np.uint64)
    print(f"{arr.sum()=}")


if __name__ == "__main__":
    main()
