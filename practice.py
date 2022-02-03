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


def get_comb(s, n):
    res = []
    if n == 0:
        return [[]]

    for i in range(len(s)):
        elem = s[i]
        for C in get_comb(s[i+1:], n-1):
            res.append([elem] + C)

    return res


def gcd(a, b):
    for i in range(min(a, b), -1, -1):
        if a % i == 0 and b % i == 0:
            return i

    return 1


T = int(input())
for _ in range(T):
    seq = list(map(int, input().split()))
    N, s = seq[0], seq[1:]

    sub = 0
    temp = list(combinations(s, 2))
    for tt in temp:
        sub += gcd(tt[0], tt[1])

    print(sub)