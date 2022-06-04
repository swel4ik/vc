from threading import Lock


class FunctionServingWrapper(object):
    """
    Class of wrapper for restriction count of simultaneous function calls
    """
    def __init__(self,
                 callable_function: callable,
                 count_of_parallel_users: int = 8):
        self.callable_function = callable_function
        self.resources = [Lock() for _ in range(count_of_parallel_users)]
        self.call_mutex = Lock()

    def __call__(self, *args, **kwargs):
        """
        Run call method of target callable function
        Args:
            *args: args for callable function
            **kwargs: kwargs for callable function
        Returns:
            Return callable function results
        """
        self.call_mutex.acquire()
        i = -1
        while True:
            for k in range(len(self.resources)):
                if not self.resources[k].locked():
                    i = k
                    break
            if i > -1:
                break

        self.resources[i].acquire()
        self.call_mutex.release()

        result = self.callable_function(*args, **kwargs)
        self.resources[i].release()

        return result
