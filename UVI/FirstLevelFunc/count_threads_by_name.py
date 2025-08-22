import threading

def count_threads_by_name(target_name='MainCode'):
    count = 0
    for thread in threading.enumerate():
        if thread.name == target_name:
            count += 1
    return count
