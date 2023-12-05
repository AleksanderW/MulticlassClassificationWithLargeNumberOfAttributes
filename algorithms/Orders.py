def lex1_order(aggregated):
    """
    Sorts the aggregated data based on the first and second elements of each tuple, in descending order.
    If there are ties, it uses the index of the tuple as the tiebreaker.

    Args:
        aggregated (List[List[Tuple[int, int]]]): The aggregated data to be sorted.

    Returns:
        List[List[Tuple[int, int, int]]]: The sorted aggregated data.
    """
    result = []
    for row in aggregated:
        rows = sorted(
            [(tup[0], tup[1], i) for i, tup in enumerate(row)],
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        result.append(rows)
    return result


# def lex1_order(aggregated):
#     """
#     Sorts the aggregated data based on the first and second elements of each tuple, in descending order.
#     If there are ties, it uses the index of the tuple as the tiebreaker.
#
#     Args:
#         aggregated (List[List[Tuple[int, int]]]): The aggregated data to be sorted.
#
#     Returns:
#         List[List[Tuple[int, int, int]]]: The sorted aggregated data.
#     """
#     result = []
#     for row in aggregated:
#         rows = [(tup[0], tup[1], i) for i, tup in enumerate(row)]
#         # Sort the list of tuples based on the first and second elements, in descending order
#         rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
#         result.append(rows)
#     return result


def lex2_order(aggregated):
    """
    Sorts the aggregated data in lexicographical order.

    Args:
        aggregated (List[List[Tuple]]): The aggregated data to be sorted.

    Returns:
        List[List[Tuple]]: The sorted aggregated data.
    """
    # Use a list comprehension to sort each row in the aggregated data
    result = [[(tup[0], tup[1], i) for i, tup in enumerate(row)] for row in aggregated]

    # Sort the modified tuples based on the specified key using a lambda function
    result = [sorted(row, key=lambda x: (-x[1], -x[0], x[2])) for row in result]

    return result


# def lex2_order(aggregated):
#     """
#     Sorts the aggregated data in lexicographical order.
#
#     Args:
#         aggregated (List[List[Tuple]]): The aggregated data to be sorted.
#
#     Returns:
#         List[List[Tuple]]: The sorted aggregated data.
#     """
#     # Create an empty list to store the sorted aggregated data
#     result = []
#
#     # Iterate over each row in the aggregated data
#     for row in aggregated:
#         # Create a list comprehension to modify each tuple in the row by appending the index value
#         rows = [(tup[0], tup[1], i) for i, tup in enumerate(row)]
#
#         # Sort the modified tuples based on the specified key
#         rows.sort(key=lambda x: (-x[1], -x[0], x[2]))
#
#         # Append the sorted modified tuples to the result list
#         result.append(rows)
#
#     # Return the sorted aggregated data
#     return result


def xu_yager_order(aggregated):
    """
    Sorts the aggregated data in descending order based on the sum of the first two elements of each tuple,
    then by the difference between the second and first elements, and finally by the index of the tuple.

    Parameters:
    - aggregated: A list of lists where each inner list contains tuples of floats.

    Returns:
    - result: A list of lists where each inner list contains tuples of floats and an integer index.
    """

    result = [
        [(round(sum(tup), 5), tup[1] - tup[0], i) for i, tup in enumerate(row)]
        for row in aggregated
    ]
    result = [sorted(rows, key=lambda x: (-x[0], -x[1], x[2])) for rows in result]
    return result


# def xu_yager_order(aggregated):
#     result = []
#     for row in aggregated:
#         rows = [(round(sum(tup), 5), round(tup[1] - tup[0], 5), i) for i, tup in enumerate(row)]
#         rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
#         result.append(rows)
#     return result


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
    set = [
        [
            (0.4, 0.4),
            (1.0, 1.0),
            (0.3, 0.4),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.2, 0.4),
            (1.0, 1.0),
            (0.3, 0.3),
            (0.4, 0.4),
        ],
        [
            (0.4, 0.4),
            (1.0, 1.0),
            (0.3, 0.4),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.2, 0.4),
            (1.0, 1.0),
            (0.3, 0.3),
            (0.4, 0.4),
        ],
    ]
    print("xuyager", get_decisions(xu_yager_order(set)))
    print("lex2", get_decisions(lex1_order(set)))
    print("lex2", get_decisions(lex2_order(set)))
