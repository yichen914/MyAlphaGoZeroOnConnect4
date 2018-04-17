import numpy as np

from multiprocessing.pool import ThreadPool


def format_state(state, env):
    return np.reshape(state, (1, env.width, env.height, 1))


def exec_by_threadpool(method, args_tuple_list, pool_size=5):
    pool = ThreadPool(pool_size)
    results = []
    for i in range(len(args_tuple_list)):
        results.append(pool.apply_async(method, args=args_tuple_list[i]))
    pool.close()
    pool.join()
    results = [r.get() for r in results]
    return results


