import sys


if len(sys.argv) == 1:
    print("meows")
elif len(sys.argv) == 3 and sys.argv[1] == "-n":
    n = int(sys.argv[2])
    for _ in range(n):
        print("meow")
else:
    print("usage: meows2.py")


#* enter this on terminal `python meows2.py -n 2`