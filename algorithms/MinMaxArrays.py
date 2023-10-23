def find_min_max_pairs(subtable):
    min_max_pairs = []
    for i in range(len(subtable[0])):
        min_max_row = []
        for j in range(len(subtable[0][0])):
            min_prob = min(knn[i][j] for knn in subtable)
            max_prob = max(knn[i][j] for knn in subtable)
            min_max_pair = (min_prob, max_prob)
            min_max_row.append(min_max_pair)
        min_max_pairs.append(min_max_row)

    return min_max_pairs
