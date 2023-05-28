import threading


class Task:
    def __init__(self, code):
        self.code = code
        self.result = {}
        self.lock = threading.Condition()

    def get_result(self):
        with self.lock:
            self.lock.wait()
        return self.result

    def set_result(self, result_dic):
        with self.lock:
            self.result = result_dic
            self.lock.notify()