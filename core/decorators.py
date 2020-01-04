'''
File containing any useful custom function decorators.
TODO - replace print statements with logs.
'''

def timer(f):
    '''
    Logs entry, exit, and time spent in a function.
    To be used as a decorator.
    '''
    def g(*args, **kwargs):
        print(f'Starting {f.__name__}')
        # logging.info(f'TIMER: [{f.__name__}()] started')

        start = time.perf_counter()
        result = f(*args, **kwargs)

        print(f'TIMER: [{f.__name__}()] ended.    Time elapsed: {time.perf_counter() - start} seconds')
        # logging.info(f'TIMER: [{f.__name__}()] ended.    Time elapsed: {time.perf_counter() - start} seconds')
        return result

    return g
