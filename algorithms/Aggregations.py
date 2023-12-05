import numpy as np


def A1_aggr(min_max_array):
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    mean_min_max = np.zeros((num_rows, num_cols, 2))

    for subtable in min_max_array:
        mean_min_max += subtable

    mean_min_max /= num_subtables

    return mean_min_max.tolist()


def A2_aggr(min_max_array):
    mean_min_max = []
    for i in range(len(min_max_array[0])):
        mean_row = []
        for j in range(len(min_max_array[0][0])):
            sum_min = 0
            max_array = []
            for subtable in min_max_array:
                sum_min += subtable[i][j][0]
                for l in range(len(min_max_array)):
                    new_row = []
                    for m in range(len(min_max_array)):
                        if l == m:
                            new_row.append(min_max_array[m][i][j][0])
                        else:
                            new_row.append(min_max_array[m][i][j][1])
                    max_array.append(sum(new_row) / len(min_max_array))
            mean_min = sum_min / len(min_max_array)
            mean_row.append((mean_min, max(max_array)))
        mean_min_max.append(mean_row)
    return mean_min_max


def A3_aggr(min_max_array):
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    mean_min_max = []
    for i in range(num_rows):
        mean_row = []
        for j in range(num_cols):
            sum_min = sum(subtable[i][j][0] for subtable in min_max_array)
            sum_max_sq = sum(subtable[i][j][1] ** 2 for subtable in min_max_array)
            sum_max = sum(subtable[i][j][1] for subtable in min_max_array)

            mean_min = sum_min / num_subtables
            mean_max = sum_max_sq / sum_max if sum_max != 0 and not np.isnan(sum_max_sq) else 0

            mean_row.append((mean_min, mean_max))
        mean_min_max.append(mean_row)

    return mean_min_max


def A4_aggr(min_max_array):
    mean_min_max = []
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    for i in range(num_rows):
        mean_row = []
        for j in range(num_cols):
            sum_min = 0
            sum_max_cb = 0
            sum_max_sq = 0
            for subtable in min_max_array:
                sum_min += subtable[i][j][0]
                sum_max_cb += subtable[i][j][1] ** 3
                sum_max_sq += subtable[i][j][1] ** 2
            mean_min = sum_min / num_subtables
            if sum_max_sq != 0 and not np.isnan(sum_max_cb):
                mean_max = sum_max_cb / sum_max_sq
            else:
                mean_max = 0
                mean_min = 0
            mean_row.append((mean_min, mean_max))
        mean_min_max.append(mean_row)
    return mean_min_max


def A5_aggr(min_max_array):
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])
    mean_min_max = []

    for i in range(num_rows):
        mean_row = []
        for j in range(num_cols):
            sum_min_sq = sum(subtable[i][j][0] ** 2 for subtable in min_max_array)
            sum_max_cb = sum(subtable[i][j][1] ** 3 for subtable in min_max_array)

            sqrt_mean_min = np.sqrt(sum_min_sq / num_subtables)
            cbrt_mean_max = np.cbrt(sum_max_cb / num_subtables)

            mean_row.append((sqrt_mean_min, cbrt_mean_max))
        mean_min_max.append(mean_row)

    return mean_min_max


def A6_aggr(min_max_array):
    mean_min_max = []
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    for i in range(num_rows):
        mean_row = []
        for j in range(num_cols):
            sum_min_cb = 0
            sum_max_qd = 0
            for subtable in min_max_array:
                sum_min_cb += subtable[i][j][0] ** 3
                sum_max_qd += subtable[i][j][1] ** 4
            cbrt_mean_min = np.cbrt(sum_min_cb / num_subtables)
            qdrt_mean_max = np.power(sum_max_qd / num_subtables, 1 / 4)
            mean_row.append((cbrt_mean_min, qdrt_mean_max))
        mean_min_max.append(mean_row)

    return mean_min_max


def A7_aggr(min_max_array):
    mean_min_max = []
    for i in range(len(min_max_array[0])):
        mean_row = []
        for j in range(len(min_max_array[0][0])):
            sum_max = 0
            min_array = []
            for subtable in min_max_array:
                sum_max += subtable[i][j][1]
                for l in range(len(min_max_array)):
                    new_row = []
                    for m in range(len(min_max_array)):
                        if l == m:
                            new_row.append(min_max_array[m][i][j][1])
                        else:
                            new_row.append(min_max_array[m][i][j][0])
                    min_array.append(sum(new_row) / len(min_max_array))
            mean_max = sum_max / len(min_max_array)
            mean_row.append((min(min_array), mean_max))
        mean_min_max.append(mean_row)
    return mean_min_max


def A8_aggr(min_max_array):
    mean_min_max = []
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    for i in range(num_rows):
        mean_row = []
        for j in range(num_cols):
            prod_min = 1
            sum_max_2 = 0
            sum_max = 0
            for subtable in min_max_array:
                prod_min *= subtable[i][j][0] ** 2
                sum_max_2 += subtable[i][j][1] ** 2
                sum_max += subtable[i][j][1]
            n_root_prod = np.power(prod_min, num_subtables)
            if sum_max != 0 and not np.isnan(sum_max_2):
                mean_max = sum_max_2 / sum_max
            else:
                mean_max = 0
                n_root_prod = 0
            mean_row.append((n_root_prod, mean_max))
        mean_min_max.append(mean_row)
    return mean_min_max


def A9_aggr(min_max_array):
    mean_min_max = []
    num_subtables = len(min_max_array)
    num_rows = len(min_max_array[0])
    num_cols = len(min_max_array[0][0])

    for i in range(num_rows):
        mean_row = []
        sum_min_sq = np.zeros(num_cols)
        sum_max_cb = np.zeros(num_cols)
        sum_max_sq = np.zeros(num_cols)

        for j in range(num_cols):
            for subtable in min_max_array:
                sum_min_sq[j] += subtable[i][j][0] ** 2
                sum_max_cb[j] += subtable[i][j][1] ** 3
                sum_max_sq[j] += subtable[i][j][1] ** 2

            sqrt_mean_min = np.sqrt(sum_min_sq[j] / num_subtables)
            if sum_max_sq[j] != 0 and not np.isnan(sum_max_cb[j]):
                mean_max = sum_max_cb[j] / sum_max_sq[j]
            else:
                mean_max = 0
                sqrt_mean_min = 0

            mean_row.append((sqrt_mean_min, mean_max))

        mean_min_max.append(mean_row)

    return mean_min_max


def A10_aggr(min_max_array):
    mean_min_max = []
    for i in range(len(min_max_array[0])):
        mean_row = []
        for j in range(len(min_max_array[0][0])):
            sum_min_sq = 0
            sum_max_sq = 0
            for subtable in min_max_array:
                sum_min_sq += subtable[i][j][0] ** 2
                sum_max_sq += subtable[i][j][1] ** 2
            sqrt_mean_min = np.sqrt(sum_min_sq / len(min_max_array))
            sqrt_mean_max = np.sqrt(sum_max_sq / len(min_max_array))
            mean_row.append((sqrt_mean_min, sqrt_mean_max))
        mean_min_max.append(mean_row)
    return mean_min_max

# def A1_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min = 0
#             sum_max = 0
#             for subtable in min_max_array:
#                 sum_min += subtable[i][j][0]
#                 sum_max += subtable[i][j][1]
#             mean_min = sum_min / len(min_max_array)
#             mean_max = sum_max / len(min_max_array)
#             mean_row.append((mean_min, mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A2_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min = 0
#             max_array = []
#             for subtable in min_max_array:
#                 sum_min += subtable[i][j][0]
#                 for l in range(len(min_max_array)):
#                     new_row = []
#                     for m in range(len(min_max_array)):
#                         if l == m:
#                             new_row.append(min_max_array[m][i][j][0])
#                         else:
#                             new_row.append(min_max_array[m][i][j][1])
#                     max_array.append(sum(new_row) / len(min_max_array))
#             mean_min = sum_min / len(min_max_array)
#             mean_row.append((mean_min, max(max_array)))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A3_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min = 0
#             sum_max_sq = 0
#             sum_max = 0
#             for subtable in min_max_array:
#                 sum_min += subtable[i][j][0]
#                 sum_max_sq += subtable[i][j][1] ** 2
#                 sum_max += subtable[i][j][1]
#             mean_min = sum_min / len(min_max_array)
#             if sum_max != 0 and not np.isnan(sum_max_sq):
#                 mean_max = sum_max_sq / sum_max
#             else:
#                 mean_max = 0
#                 mean_min = 0
#             mean_row.append((mean_min, mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A4_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min = 0
#             sum_max_cb = 0
#             sum_max_sq = 0
#             for subtable in min_max_array:
#                 sum_min += subtable[i][j][0]
#                 sum_max_cb += subtable[i][j][1] ** 3
#                 sum_max_sq += subtable[i][j][1] ** 2
#             mean_min = sum_min / len(min_max_array)
#             if sum_max_sq != 0 and not np.isnan(sum_max_cb):
#                 mean_max = sum_max_cb / sum_max_sq
#             else:
#                 mean_max = 0
#                 mean_min = 0
#             mean_row.append((mean_min, mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A5_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min_sq = 0
#             sum_max_cb = 0
#             for subtable in min_max_array:
#                 sum_min_sq += subtable[i][j][0] ** 2
#                 sum_max_cb += subtable[i][j][1] ** 3
#             sqrt_mean_min = np.sqrt(sum_min_sq / len(min_max_array))
#             cbrt_mean_max = np.cbrt(sum_max_cb / len(min_max_array))
#             mean_row.append((sqrt_mean_min, cbrt_mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A6_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min_cb = 0
#             sum_max_qd = 0
#             for subtable in min_max_array:
#                 sum_min_cb += subtable[i][j][0] ** 3
#                 sum_max_qd += subtable[i][j][1] ** 4
#             cbrt_mean_min = np.cbrt(sum_min_cb / len(min_max_array))
#             qdrt_mean_max = np.power(sum_max_qd / len(min_max_array), 1 / 4)
#             mean_row.append((cbrt_mean_min, qdrt_mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A7_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_max = 0
#             min_array = []
#             for subtable in min_max_array:
#                 sum_max += subtable[i][j][1]
#                 for l in range(len(min_max_array)):
#                     new_row = []
#                     for m in range(len(min_max_array)):
#                         if l == m:
#                             new_row.append(min_max_array[m][i][j][1])
#                         else:
#                             new_row.append(min_max_array[m][i][j][0])
#                     min_array.append(sum(new_row) / len(min_max_array))
#             mean_max = sum_max / len(min_max_array)
#             mean_row.append((min(min_array), mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A8_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             prod_min = 1
#             sum_max_2 = 0
#             sum_max = 0
#             for subtable in min_max_array:
#                 prod_min *= subtable[i][j][0] ** 2
#                 sum_max_2 += subtable[i][j][1] ** 2
#                 sum_max += subtable[i][j][1]
#             n_root_prod = np.power(prod_min, len(min_max_array))
#             if sum_max != 0 and not np.isnan(sum_max_2):
#                 mean_max = sum_max_2 / sum_max
#             else:
#                 mean_max = 0
#                 n_root_prod = 0
#             mean_row.append((n_root_prod, mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A9_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min_sq = 0
#             sum_max_cb = 0
#             sum_max_sq = 0
#             for subtable in min_max_array:
#                 sum_min_sq += subtable[i][j][0] ** 2
#                 sum_max_cb += subtable[i][j][1] ** 3
#                 sum_max_sq += subtable[i][j][1] ** 2
#             sqrt_mean_min = np.sqrt(sum_min_sq / len(min_max_array))
#             if sum_max_sq != 0 and not np.isnan(sum_max_cb):
#                 mean_max = sum_max_cb / sum_max_sq
#             else:
#                 mean_max = 0
#                 sqrt_mean_min = 0
#             mean_row.append((sqrt_mean_min, mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
#
#
# def A10_aggr(min_max_array):
#     mean_min_max = []
#     for i in range(len(min_max_array[0])):
#         mean_row = []
#         for j in range(len(min_max_array[0][0])):
#             sum_min_sq = 0
#             sum_max_sq = 0
#             for subtable in min_max_array:
#                 sum_min_sq += subtable[i][j][0] ** 2
#                 sum_max_sq += subtable[i][j][1] ** 2
#             sqrt_mean_min = np.sqrt(sum_min_sq / len(min_max_array))
#             sqrt_mean_max = np.sqrt(sum_max_sq / len(min_max_array))
#             mean_row.append((sqrt_mean_min, sqrt_mean_max))
#         mean_min_max.append(mean_row)
#     return mean_min_max
if __name__ == "__main__":
    tab1 = [[[(0.0, 0.6)]], [[(0.1, 0.3)]], [[(0.2, 0.7)]]]
    tab2 = [[[(0, 0)]], [[(0, 0)]], [[(0, 0)]]]
    tab3 = [[[(0, 0)]], [[(0, 0)]], [[(0.2, 0.5)]]]
    a = A1_aggr(tab1)
    b = A2_aggr(tab1)
    c1 = A3_aggr(tab1)
    c2 = A3_aggr(tab2)
    c3 = A3_aggr(tab3)
    d1 = A4_aggr(tab1)
    d2 = A4_aggr(tab2)
    d3 = A4_aggr(tab3)
    e = A5_aggr(tab1)
    f = A6_aggr(tab1)
    g = A7_aggr(tab1)
    h1 = A8_aggr(tab1)
    h2 = A8_aggr(tab2)
    h3 = A8_aggr(tab3)
    i1 = A9_aggr(tab1)
    i2 = A9_aggr(tab2)
    i3 = A9_aggr(tab3)
    j = A10_aggr(tab1)

    print('A1=', a)
    print('A2=', b)
    print('A3_1=', c1)
    print('A3_2=', c2)
    print('A3_3=', c3)
    print('A4_1=', d1)
    print('A4_2=', d2)
    print('A4_3=', d3)
    print('A5=', e)
    print('A6=', f)
    print('A7=', g)
    print('A8_1=', h1)
    print('A8_2=', h2)
    print('A8_3=', h3)
    print('A9_1=', i1)
    print('A9_2=', i2)
    print('A9_3=', i3)
    print('A10=', j)
