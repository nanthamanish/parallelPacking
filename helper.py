
import os

def read_ints(f) -> list[int]:
    return [int(x) for x in f.readline().split()]


def read_floats(f) -> list[float]:
    return [float(x) for x in f.readline().split()]

def shift_range(x, lb, ub):
    res = lb + (ub - lb) * x
    return res

def to_str(l):
    return ' '.join(map(str, l)) + '\n'

def make_dir(cwd, new_dir):
    new_dir = "{cwd}\\{nd}".format(cwd=cwd, nd=new_dir)
    if os.path.isdir(new_dir) is False:
        os.makedirs(new_dir)
        print("Folder {id} created".format(id = new_dir))