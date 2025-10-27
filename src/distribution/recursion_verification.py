import numpy as np

if __name__ == '__main__':
    L = 10

    var_y1 = 2

    var_w = 1
    var_b = 4 * 1
    n_l = 1

    prev_var_yl = var_y1

    count = 0

    for i in range(1, L + 1):
        count += 1

        var_yl = n_l / 2 * var_w * prev_var_yl + var_b

        prev_var_yl = var_yl

        a = np.prod([n_l / 2 * var_w] * (i))
        a1 = a * var_y1

        print(var_yl, var_yl - a1, a1)
    print("FINISHED", count)

    recursive_var_yl = var_yl

    for L in range(1, L + 1):

        a = np.prod([n_l / 2 * var_w] * (L))
        a1 = a * var_y1

        b = var_b

        if L >= 2:
            count = 0

            for l in range(1, L + 1):
                count += 1

                b_sub = [n_l / 2 * var_w] * (L - l + 1)
                b_sub_prod = np.prod(b_sub)

                b += (b_sub_prod * var_b)
            print(L, count)

        theoretical = a1 + b

        print(theoretical, b, a1)
        # print(recursive_var_yl - theoretical)
