import time
import threading

import pympler.tracker

d_objs = [] 
class PerpetualTimer():
   
    def __init__(self, time_to_wait, callback):
       self.time_to_wait = time_to_wait
       self.callback = callback
       self.thread = threading.Timer(self.time_to_wait, self.process_callback)

    def process_callback(self):
       self.callback()
       self.thread = threading.Timer(self.time_to_wait, self.process_callback)
       self.thread.start()

    def start(self):
       self.thread.start()

    def cancel(self):
       self.thread.cancel()


def memory_report():
    summary_tracker = pympler.tracker.SummaryTracker()
    summary_tracker.print_diff() 

if __name__ == '__main__':
    
    summary_reporter = PerpetualTimer(1.0, memory_report)
    summary_reporter.start()
    
    while True:
        d = {}
        d['foo'] = 'ass'
        d['time'] = time.time()
        d_objs.append(d)