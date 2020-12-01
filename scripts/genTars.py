
import os

dd = "D:/Datasets/MyTars/"

f = open(dd + "random", "wb+")
f.write(bytes(os.urandom(100000)))
f.close()
    
