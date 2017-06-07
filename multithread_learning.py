import threading
import time

exitFlag = 0


class myThread(threading.Thread):  # inherit parent class threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):  # after creating a new thread it call function 'run' directly
        print "Starting " + self.name
        print_time(self.name, self.counter, 5)
        print "Exiting " + self.name


def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threading.Thread.exit()
        time.sleep(delay)
        print "%s: %s" % (threadName, time.ctime(time.time()))
        counter -= 1


# create new thread
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)
thread3 = myThread(3, "Thread-3", 2)
thread4 = myThread(4, "Thread-4", 4)
thread5 = myThread(5, "Thread-5", 3)
thread6 = myThread(6, "Thread-6", 1)
thread7 = myThread(7, "Thread-7", 1)
thread8 = myThread(8, "Thread-8", 1)
thread9 = myThread(9, "Thread-9", 1)

# thread start
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()

print "Exiting Main Thread"