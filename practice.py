from itertools import combinations
from sys import setrecursionlimit
setrecursionlimit(10 ** 6)

# 중복조합
# types = int(input())
# N = int(input())


def choose(n, k):
    if k == 0:
        return 1
    elif n < k:
        return 0
    else:
        return choose(n-1, k-1) + choose(n-1, k)


while True:
    ks = list(map(int, input().split()))
    k = ks[0]
    if k == 0:
        break
    s = ks[1:]

    ans = combinations(s, 6)
    for a in ans:
        print(" ".join(map(str, a)))

    print()


