import os
import sys
import time
import unittest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from connect_four_env import ConnectFourEnv

def timer(f):
    def g(*args, **kwargs):
        print(f'Starting {f.__name__}')
        start = time.perf_counter()
        result = f(*args, **kwargs)
        print(f'TIMER: [{f.__name__}()] ended.    Time elapsed: {time.perf_counter() - start} seconds')
        return result
    return g


class TestNewState(unittest.TestCase):

    def test_step():

        connect_four_env = ConnectFourEnv()

    @timer
    def test_upper(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()