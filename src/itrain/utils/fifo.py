from queue import Queue

class FIFO:
    def __init__(self, maxsize):
        self.q = Queue(maxsize=maxsize)
    
    def put(self, item):
        jtem = None
        if self.q.full():
            jtem = self.q.get()
        self.q.put(item)

        return jtem
    
    def get(self):
        if self.q.empty():
            return None
        return self.q.get()
    
    def size(self):
        return self.q.qsize()
