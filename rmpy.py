import os 
import sys

py_list = []
os.system("ps ux | grep 'python -u experiment.py' > py_code.txt")
for i in open("py_code.txt"):
    py_list.append(i.strip())
for i in py_list:
    if float(i.split()[2]) > 1.0:   
        print("kill -9 %s" % i) 
        print("kill -9 %s" % i.split()[1])
        os.system("kill -9 %s" % i.split()[1])
os.system("rm py_code.txt")
    # print("kill -9 %s" % i.split()[1])
    # os.system("kill -9 %s" % i.split()[2])
    # print("kill -9 %s" % i.split()[2])
# for i in py_code.split('\n'):
#     py_list.append(i)
# for i in py_list:
#     # if i != '':
#         # os.system("kill -9 %s" % i)
#     print("kill %s" % i)