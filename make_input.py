import random
from helper import read_floats, read_ints, to_str, shift_range, make_dir
import sys
import os

DENSITY_LOWER_BOUND = 1
DENSITY_UPPER_BOUND = 1
DESTINATIONS = 10
WTPACK_PROBLEM_CNT = 100
PRECISION = 3
SEED = 1


def main():
    random.seed(SEED)
    #inp = input("Enter File Name (without extension): ")

    make_dir(os.getcwd(), "input")

    inp = sys.argv[1]
    fname = "wtpack/{}.txt".format(inp)
    print(fname)
    f_in = open(fname, "r")

    for p in range(WTPACK_PROBLEM_CNT):
        cont_dim = read_ints(f_in)
        n_packs, vol = read_floats(f_in)
        n_packs = int(n_packs)
        t_boxes = 0
        out_fname = "input/{inp}_{prob}.txt".format(inp=inp, prob=p)
        f_out = open(out_fname, "w")

        id = 0
        items = []
        for i in range(n_packs):
            l, lo, b, bo, h, ho, numBox, wt, lbear, bbear, hbear = read_floats(
                f_in)

            l, lo, b, bo, h, ho, numBox = [
                int(x) for x in [l, lo, b, bo, h, ho, numBox]]

            t_boxes += numBox

            for j in range(numBox):
                dest = random.randint(1, DESTINATIONS)

                x = random.random()
                density = shift_range(
                    x, DENSITY_LOWER_BOUND, DENSITY_UPPER_BOUND)

                # print(x, density)

                wt_upd = round(density * wt, PRECISION)
                lbear_upd = round(density * lbear, PRECISION)
                bbear_upd = round(density * bbear, PRECISION)
                hbear_upd = round(density * hbear, PRECISION)

                # wt_upd = 0

                item = [id, dest, wt_upd, l, b, h, lo, bo,
                        ho, lbear_upd, bbear_upd, hbear_upd]

                items.append(item)
                id += 1

        cont_dim.append(t_boxes)
        contStr = to_str(cont_dim)
        f_out.write(contStr)

        for item in items:
            out_str = to_str(item)
            f_out.write(out_str)

        f_out.close()
    f_in.close()

    print("Inputs Generated")


if __name__ == "__main__":
    main()
