from multiprocessing import Process, Queue
import os, time, random

def write(q):
    print 'process1 start'
    acount = q.get(True)
    for i in range(10):
        time.sleep(random.random())
        acount += 3
        print('here is process1 a = %s' %(acount))
    q.put(acount)

def read(q):
    print 'process2 start'
    a = q.get(True)
    for i in range(10):
        time.sleep(random.random())
        a -= 3
        print('here is process2 a = %s' %(a))
    q.put(a)

q = Queue()
acount = 0
q.put(acount)
pw = Process(target=write, args=(q,))
pr = Process(target=read, args=(q,))
pw.start()
pr.start()
pw.join()
pr.join()
while q.empty() == False:
    print q.get()
