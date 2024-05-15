import time


def calculate_time(func):
    def timing(*args, **kwargs):
        t1 = time.time()
        outputs = func(*args, **kwargs)
        t2 = time.time()
        print(f"Time: {(t2-t1):.3f}s")
        return outputs

    return timing


def modify_days_to_3digits(day=str):
    words = day.split()
    try:
        nday = int(words[0])
        return " ".join([f"{nday:03}"] + words[1:])
    except ValueError:
        return "9999"
