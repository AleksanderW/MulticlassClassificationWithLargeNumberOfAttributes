def lex1_order(aggregated):
    result = []
    for row in aggregated:
        rows = []
        for i, tup in enumerate(row):
            rows.append((tup[0], tup[1], i))
        rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
        result.append(rows)
    return result


def lex2_order(aggregated):
    result = []
    for row in aggregated:
        rows = []
        for i, tup in enumerate(row):
            rows.append((tup[0], tup[1], i))
        rows.sort(key=lambda x: (-x[1], -x[0], x[2]))
        result.append(rows)
    return result


def xu_yager_order(aggregated):
    result = []
    for row in aggregated:
        rows = []
        for i, tup in enumerate(row):
            rows.append((round(sum(tup), 5), round(tup[1] - tup[0], 5), i))
        rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
        result.append(rows)
    return result


# def check_if_equal_tuples(tuples, order): # Dla wszystkich
#     equal_tuples = []
#     for tup1, tup2 in combinations(tuples, 2):
#         if tup1[0] == tup2[0] and tup1[1] == tup2[1]:
#             if not tup1 in equal_tuples:
#                 equal_tuples.append(tup1)
#             if not tup2 in equal_tuples:
#                 equal_tuples.append(tup2)
#     if equal_tuples:
#         message = f"Order {order}, " + "equal tuples: " + f"{equal_tuples}"
#         warnings.warn(message, category=Warning)
#     return equal_tuples
# def xu_yager_order_testing(aggregated):
#     decisions = []
#     for row in aggregated:
#         result = []
#         for i, tup in enumerate(row):
#             result.append((round(sum(tup), 5), round(tup[1] - tup[0], 5), i))  # (SUMA, SZEROKOŚĆ, INDEX)
#         result.sort(key=lambda x: (-x[0], -x[1], x[2]))  # sort (sum_desc, width_desc, index_asc)
#         decisions.append(result[0][2])
#         print("Sorted:\t", *(f"({r[0]}, {r[1]}, {r[2]})" for r in result))
#         check_if_equal_tuples(result, "XuYager")
#     return decisions
#
# def lex1_order_testing(aggregated):
#     decisions = []
#     for row in aggregated:
#         result = []
#         for i, tup in enumerate(row):
#             result.append((tup[0], tup[1], i))
#         result.sort(key=lambda x: (-x[0], -x[1], x[2]))
#         decisions.append(result[0][2])
#         print("Sorted:\t", *(f"({r[0]}, {r[1]}, {r[2]})" for r in result))
#         check_if_equal_tuples(result, "lex1")
#     return decisions
#
#
# def lex2_order_testing(aggregated):
#     decisions = []
#     for row in aggregated:
#         result = []
#         for i, tup in enumerate(row):
#             result.append((tup[0], tup[1], i))
#         result.sort(key=lambda x: (-x[1], -x[0], x[2]))
#         decisions.append(result[0][2])
#         print("Sorted:\t", *(f"({r[0]}, {r[1]}, {r[2]})" for r in result))
#         check_if_equal_tuples(result, "lex2")
#     return decisions


def get_decisions(data):
    return [tup[0][2] for tup in data]


if __name__ == "__main__":
    print("XUYAGER")
    aggregated1 = [[(0.0, 0.3), (0.1, 0.2)]]
    aggregated2 = [[(0.1, 0.2), (0.0, 0.3)]]
    print("Decision", get_decisions(xu_yager_order(aggregated1)))
    print("Decision", get_decisions(xu_yager_order(aggregated2)))

    aggregated3 = [[(0.2, 0.5), (0.2, 0.4)]]
    aggregated4 = [[(0.0, 0.2), (0.1, 0.2)]]
    print("\nLEX1")
    print("Decision", get_decisions(lex1_order(aggregated1)))
    print("Decision", get_decisions(lex1_order(aggregated2)))
    print("Decision", get_decisions(lex1_order(aggregated3)))
    print("Decision", get_decisions(lex1_order(aggregated4)))

    print("\nLEX2")
    print("Decision", get_decisions(lex2_order(aggregated1)))
    print("Decision", get_decisions(lex2_order(aggregated2)))
    print("Decision", get_decisions(lex2_order(aggregated3)))
    print("Decision", get_decisions(lex2_order(aggregated4)))

    print("\nTest IWIFSGN2022")
    set = [[(0.5, 0.8), (0.2, 0.9), (1.0, 1.0)]]
    print("lex1", get_decisions(lex1_order(set)))
    print("lex2", get_decisions(lex2_order(set)))

    print("\nTest equality")
    set = [[(0.4, 0.4), (1.0, 1.0), (0.3, 0.4), (1.0, 1.0), (0.0, 1.0), (0.2, 0.4), (1.0, 1.0), (0.3, 0.3), (0.4, 0.4)],
           [(0.4, 0.4), (1.0, 1.0), (0.3, 0.4), (1.0, 1.0), (0.0, 1.0), (0.2, 0.4), (1.0, 1.0), (0.3, 0.3), (0.4, 0.4)]]
    print("xuyager", get_decisions(xu_yager_order(set)))
    print("lex2", get_decisions(lex1_order(set)))
    print("lex2", get_decisions(lex2_order(set)))
