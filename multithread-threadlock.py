import threading
import time

count = 0

class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print "Starting " + self.name
        # require lock return True if get it
        # param timeout is optional if ignore it will block until the thread get the lock
        # if set timeout if will return False if timeout
        global count
        threadLock.acquire()
        print(count)
        temp = print_time(self.name, self.counter, 3)
        count += temp
        print('Exiting' + self.name)
        print(count)
        # release the lock
        threadLock.release()


def print_time(threadName, delay, counter):
    count = 0
    while counter:
        time.sleep(delay)
        print "%s: %s" % (threadName, time.ctime(time.time()))
        counter -= 1
        count += delay
    return count


threadLock = threading.Lock()
threads = []

# create new thread
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# start threads
thread1.start()
thread2.start()

# adding threads into thread list
threads.append(thread1)
threads.append(thread2)

# wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"
