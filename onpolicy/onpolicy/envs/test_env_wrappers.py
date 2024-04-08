import unittest
from multiprocessing import Pipe, Process
from onpolicy.envs.env_wrappers import worker


class TestEnvWrappers(unittest.TestCase):
    def test_worker_step(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('step', 0))
        result = parent_conn.recv()
        p.join()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], object)  # replace with your observation space type
        self.assertIsInstance(result[1], float)  # replace with your reward type
        self.assertIsInstance(result[2], bool)  # replace with your done type
        self.assertIsInstance(result[3], dict)  # replace with your info type

    def test_worker_reset(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('reset',))
        result = parent_conn.recv()
        p.join()
        self.assertIsInstance(result, object)  # replace with your observation space type

    def test_worker_render(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('render', 'rgb_array'))
        result = parent_conn.recv()
        p.join()
        self.assertIsInstance(result, object)  # replace with your render output type

    def test_worker_reset_task(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('reset_task',))
        result = parent_conn.recv()
        p.join()
        self.assertIsInstance(result, object)  # replace with your observation space type

    def test_worker_close(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('close',))
        p.join()
        self.assertTrue(p.exitcode == 0)

    def test_worker_get_spaces(self):
        parent_conn, child_conn = Pipe()
        env_fn_wrapper = lambda: None  # replace with your environment function wrapper
        p = Process(target=worker, args=(child_conn, parent_conn, env_fn_wrapper))
        p.start()
        parent_conn.send(('get_spaces',))
        result = parent_conn.recv()
        p.join()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], object)  # replace with your observation space type
        self.assertIsInstance(result[1], object)  # replace with your share observation space type
        self.assertIsInstance(result[2], object)  # replace with your action space type