from multiprocessing import Process, Queue, Manager
import numpy as np

class Test:
    def __init__(self):

        self.count = 10
        self.matrix = np.arange(16).reshape(4,4)

    def testsignal(self):
        mg = Manager()
        list = mg.list()




    # def multiply(self, x, y):
    #     print("process starts")
    #     return x*y
        # q.put(x * y)
        # self.count = x*y
        # print("end")

    # def circulation_2(self):
    #     record = []
    #     q= Queue()
    #     for x in range(1,3):
    #         for y in range(1,3):
    #             p = Process(target= self.multiply, args=( x, y,))
    #             p.start()
    #             record.append(p)
    #         for p in record:
    #             p.join()
    #         record = []
    #     while(not q.empty()):
    #         print(q.get())
    #     print(self.count,'count')

if __name__ == '__main__':
    t = Test()
    a = list()
    # matrix1 = np.arange(9).reshape(3,3)
    # matrix2 = np.arange(16).reshape(4, 4)
    # a.append(matrix1)
    # a.append(matrix2)
    # print(a)





