import time


def calculate_time(name: str):
    def inner(func):
        def timing(*args, **kwargs):
            t1 = time.time()
            outputs = func(*args, **kwargs)
            t2 = time.time()

            print(f"{name} Time: {(t2-t1):.3f}s")
            return outputs

        return timing

    return inner


def describe(name: str, desciption: str):
    def inner(func):
        def decorate(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return outputs, name, desciption

        return decorate

    return inner
