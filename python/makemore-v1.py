import numpy as np
import time
import torch
import sys


def run():
    print('test')


def lesson1():
    print('Lesson 1')

    a = torch.zeros((3, 5), dtype=torch.int32)
    print(a.dtype)
    print(a)

    print('---')

    a[1][3] = 1
    print(a)

    a[1][3] += 1
    print(a)

def torch_check():
    x = torch.rand(5, 3)
    print(x)


if __name__ == '__main__':
    print(f'✅ Python Version:{sys.version}')
    print(f'✅ PyTorch Version:{torch.__version__}')
    print(f'✅ NumPy Version:{np.__version__}')
    torch_check()
    lesson1()
    start_time = time.time()
    run()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.6f} seconds")
