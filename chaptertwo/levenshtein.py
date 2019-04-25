# Dynamic Programming Solution to Minimum Edit Distance
# @author: noahshpak


def del_cost(s): return 1
def ins_cost(s): return 1
def subst_cost(s, t): return 2 if s != t else 0

# -< Algorithm >- #
# 
# Define D(i,j) as the edit distance between X[1..i] and Y[1..j]
# Thus, the edit distance between X and Y is D(n, m)
# where len(X) == n and len(Y) == m
def edit_distance(src: str, tgt: str):
    if not(isinstance(src, str) and isinstance(tgt, str)):
        raise ValueError("Inputs should be strings")

    n, m = len(src), len(tgt)
    D = [[0 for _ in range(m+1)] for _ in range(n+1)] # construct n+1 x m+1 matrix
    # Initialize small sub-problems
    # going from i -> 0 chars requires i deletes 
    for i in range(1, n+1):
        D[i][0] = D[i-1][0] + del_cost(src[i-1]) 
    for j in range(1, m+1):
        D[0][j] = D[0][j-1] + ins_cost(tgt[j-1])

    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(
                D[i-1][j] + del_cost(src[i-1]),
                D[i][j-1] + ins_cost(tgt[j-1]),
                D[i-1][j-1] + subst_cost(src[i-1], tgt[j-1])
            )
    return D[n][m]



