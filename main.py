import sys
from functools import reduce
from collections import deque

sys.setrecursionlimit(10 ** 6)  # 기본값: 1000


# 평균은 넘겠지
def average():
    case = int(input())

    for i in range(case):
        test = list(map(int, input().split(" ")))
        num_student, score = test[0], test[1:]
        avg = sum(score) / num_student

        cnt = 0
        for s in score:
            if s > avg:
                cnt += 1

        print("{:.3f}%".format(cnt / num_student * 100))


# 문자열 다루기 기본
def solution1(s):
    if len(s) == 4 or len(s) == 6:
        return s.isdigit()

    return False


# 실패율
def solution3(N, stages):
    ans = []
    fail_per = []
    for i in range(1, N + 1):
        challenger, failure = 0, 0
        for stage in stages:
            if stage >= i:
                if stage == i:
                    failure += 1
                challenger += 1

        # 예외 처리: 시도 조차 못한 경우 (실패율은 0%)
        if challenger == 0:
            fail_per.append(0)
        else:
            fail_per.append(failure / challenger)

    fail_per = list(enumerate(fail_per))
    fail_per = sorted(fail_per, key=lambda x: x[1], reverse=True)

    for idx, item in fail_per:
        ans.append(idx + 1)

    return ans


# Count Primes
# 에라스토테네스의 체
class Solution4(object):
    def countPrimes(self, n):
        result = 0
        if n <= 1:
            return 0

        primes = [True] * n
        primes[0] = primes[1] = False

        for i in range(2, n):
            if primes[i]:
                for j in range(i * 2, n, i):
                    primes[j] = False

        for prime in primes:
            if prime:
                result += 1

        return result


# 문자열 압축
def solution2(s):
    answer = [len(s)]

    for i in range(1, len(s) // 2 + 1):
        string = ""
        substring = s[:i]
        cnt = 1

        for j in range(i, len(s), i):
            if substring == s[j:j + i]:
                cnt += 1
            else:
                if cnt > 1:
                    string += str(cnt)
                string += substring
                cnt = 1
                substring = s[j:j + i]

        if cnt > 1:
            string += str(cnt) + substring
        else:
            string += substring

        answer.append(len(string))
    return min(answer)


# a = [1, 2, 3, 4]
# print(a[1:100]) : 정상동작


def singleNumber(nums):
    nums.sort()
    answer = 0
    for idx, num in enumerate(nums):
        if idx % 2 == 0:
            answer += num
        else:
            answer -= num

    return answer


def summaryRanges(nums: list[int]) -> list[str]:
    answer = []
    flag = 1
    temp = ""
    for i in range(len(nums)):
        item = str(nums[i])

        if i == len(nums) - 1:
            temp += str(nums[i])
            answer.append(temp)
            break

        if nums[i] + 1 == nums[i + 1]:
            if flag:
                temp += item + "->"
                flag = 0

        else:
            temp += item
            answer.append(temp)
            temp = ""
            flag = 1

    return answer


def validnumber(s):
    number = map(int, s.split())
    temp = []
    for num in number:
        num = num ** 2
        temp.append(num)

    return sum(temp) % 10


def save(people, limit):
    people.sort()
    boat = 0
    start = 0
    end = len(people) - 1

    while start <= end:
        if start == end:
            return boat + 1

        if people[start] + people[end] <= limit:
            boat += 1
            start += 1
            end -= 1
        else:
            boat += 1
            end -= 1

    return boat


# 한 세트: 0~9
# 6과 9는 호환 가능
def roomnumber(s):
    ans = 0
    temp = [0] * 10
    flag = 1
    for number in s:
        temp[number] += 1

    if temp[6] != temp[9]:
        diff = abs(temp[6] - temp[9])
        for i in range(diff // 2):
            if temp[6] > temp[9]:
                temp[6] -= 1
                temp[9] += 1
            else:
                temp[9] -= 1
                temp[6] += 1
    ans = max(temp)

    return ans


def treasure(a, b):
    a.sort()
    cnt = 0

    for item_a in a:
        biggest = max(b)
        cnt += item_a * biggest
        b.remove(biggest)

    return cnt


def num_digit(num):
    ans = []
    for digit in range(0, 10):
        digit = str(digit)
        ans.append(num.count(digit))
        print(ans[-1])


def passenger(stations):
    answer = []
    for station in stations:
        total = station[1] - station[0]
        if stations.index(station) == 0:
            answer.append(total)
        else:
            rest = answer[-1]
            answer.append(rest + total)

    return max(answer)


def plug(multi):
    return sum(multi) - (len(multi) - 1)


def fib(n):
    if n <= 1:
        return n

    f = [0, 1]
    for i in range(2, n + 1):
        f.append(f[i - 2] + f[i - 1])

    return f[-1]


def fee(callings):
    res = ""
    y = m = 0
    for calling in callings:
        y += (calling // 30 + 1) * 10
        m += (calling // 60 + 1) * 15

    if y < m:
        res += "Y "
    elif y > m:
        res += "M "
    else:
        res += "Y M "
    res += str(min(y, m))

    return res


def addDigits(num):
    while True:
        sums = 0
        while num:
            sums += num % 10
            num = num // 10

        if sums < 10:
            return sums
        num = sums


def twoSum(nums, target):
    # indexes = []
    # for i in range(len(nums) - 1):
    #     for j in range(i+1, len(nums)):
    #         if nums[i] + nums[j] == target:
    #             indexes.append(i)
    #             indexes.append(j)
    #
    # return indexes
    dict = {}
    for idx, num in enumerate(nums):
        dict[target - num] = idx

    for idx, num in enumerate(nums):
        if num in dict.keys() and idx != dict[num]:
            return [idx, dict[num]]


def lemonadeChange(bills):
    cash = {
        5: 0,
        10: 0,
        20: 0
    }
    for bill in bills:
        cash[bill] += 1

        if cash[5] < 1:
            return False

        if bill == 10:
            cash[5] -= 1

        elif bill == 20:
            if cash[10] > 0:
                cash[10] -= 1
                cash[5] -= 1
            elif cash[5] >= 3:
                cash[5] -= 3
            else:
                return False

    return True


def isPowerOfFour(n) -> bool:
    # return n > 0 and math.log(n, 4).is_integer()
    if n < 1:
        return False
    if n == 1:
        return True
    return isPowerOfFour(n / 4)


def isHarshad(x):
    temp = list(map(int, str(x)))
    if x % sum(temp) == 0:
        return True

    return False


def function1(numbers, k):
    temp = {}
    cnt = 0
    for idx, number in enumerate(numbers):
        temp[k - number] = idx

    for idx, number in enumerate(numbers):
        if number in temp and idx != temp[number]:
            cnt += 1
    # 중복 주의
    return cnt // 2


def invite(trees):
    day = 1
    trees.sort(reverse=True)
    grow = []
    for tree in trees:
        grow.append(day + tree)
        day += 1
    return max(grow) + 1


def radio(start, end, moves):
    paths = [abs(start - end)]
    for move in moves:
        paths.append(abs(end - move) + 1)

    return min(paths)


# def alpha_centuri(start, end):
#     cnt = 1
#     f = [-1, 0, 1]
#     while start == end - 1:
#         flag = max(f)
#         if flag + start < end - 1:
#             start += flag
#             cnt += 1
#             f = [flag - 1, flag, flag + 1]
#         else:
#             for ff in f:
#                 if ff + start == end - 1:
#                     cnt += 1
#                     break
#
#     return cnt


def door(length, first):
    if length >= 6:
        return "Love is open door"

    temp = [first, 1 - first]
    if length % 2 == 0:
        sol = temp * (length // 2)
    else:
        sol = temp * (length // 2) + temp[0]

    for s in sol[1:]:
        print(s)

    return 0


def gb_day(prob, days, cur):
    gGood, gBad, bGood, bBad = map(float, prob.split())
    good, bad = [], []

    if cur:
        good.append(bGood)
        bad.append(bBad)
    else:
        good.append(gGood)
        bad.append(gBad)

    for day in range(1, days):
        g, b = good[-1], bad[-1]
        good.append(g * gGood + b * bGood)
        bad.append(g * gBad + b * bBad)

    return str(round(good[-1] * 1000)) + "\n" + str(round(bad[-1] * 1000))


def make_big(number, k):
    number = list(map(int, list(number)))
    stack = []
    cnt = 0
    for n in number:
        if not stack:
            stack.append(n)
            continue

        elif cnt < k:
            for s in reversed(stack):
                if s or cnt < k or n > s:
                    cnt += 1
                    stack.pop()
                else:
                    break
        stack.append(n)
    if k > cnt:
        stack = stack[:-k + cnt]

    return "".join(list(map(str, stack)))


def chocolate(n, m):
    return (n - 1) + (m - 1) * n


def microwave(times):
    buttons = [300, 60, 10]
    temp = [0] * 3
    flag = 0

    while times >= 10:
        if times >= buttons[flag]:
            times -= buttons[flag]
            temp[flag] += 1
        else:
            flag += 1

    if times > 0:
        return -1

    return " ".join(map(str, temp))


def gymSuits(n, lost, reserve):
    for i in range(1, n + 1):
        if i in lost and i in reserve:
            lost.remove(i)
            reserve.remove(i)

    lost.sort()
    reserve.sort()

    for r in reserve:
        if not lost:
            break
        for lo in lost:
            if r - 1 <= lo <= r + 1:
                lost.remove(lo)
                break

    return n - len(lost)


def coin(coin_list, target):
    cnt = []
    for co in reversed(coin_list):
        if target == 0:
            break

        if co > target:
            continue
        else:
            cnt.append(target // co)
            target -= cnt[-1] * co

    return sum(cnt)


# k층 n호
def residents(k, n):
    # if k == 0:
    #     return n
    # temp = []
    # for i in range(1, n + 1):
    #     temp.append(residents(k - 1, i))
    #
    # return sum(temp)
    temp = [i for i in range(1, n + 1)]
    for i in range(k):
        for j in range(1, n):
            temp[j] = temp[j] + temp[j - 1]

    return temp[-1]


def remainder(numbers):
    temp = []
    for idx, num in enumerate(numbers):
        w = num % 42
        if w not in temp:
            temp.append(w)

    return len(temp)


def mars_math(formula):
    formula = formula.split()
    temp = float(formula[0])

    for i in range(1, len(formula)):
        if formula[i] == "@":
            temp *= 3
        elif formula[i] == "%":
            temp += 5
        elif formula[i] == "#":
            temp -= 7

    print("{:.2f}".format(temp))


def lost_paran(form):
    form = form.split("-")
    temp = []
    for idx, f in enumerate(form):
        if "+" in f:
            plus = list(map(int, f.split("+")))
            temp.append(str(sum(plus)))
        else:
            temp.append(f)

    return reduce(lambda x, y: x - y, list(map(int, temp)))


def snake_bird(length, fruits):
    fruits.sort()
    for fruit in fruits:
        if length >= fruit:
            length += 1

    return length


def partial_sum(n_length, n, target):
    # 시간초과 발생(length와 target의 범위 주목)
    # cumSum = []
    # for idx, number in enumerate(n):
    #     if not cumSum:
    #         cumSum.append(number)
    #     else:
    #         cumSum.append(cumSum[-1] + number)
    #
    # cnt = 1
    # while cnt <= length:
    #     for i in range(cnt-1, length):
    #         temp = cumSum[cnt - 1] if i == cnt-1 else cumSum[i] - cumSum[i-cnt]
    #         if target <= temp:
    #             return cnt
    #     cnt += 1
    #
    # return 0
    start, end = 0, 0
    temp = n[0]
    answer = 100001
    while True:
        if temp >= target:
            answer = min(answer, end - start + 1)
            temp -= n[start]
            start += 1
        else:
            end += 1
            if end == n_length:
                break
            temp += n[end]

    return 0 if answer == 100001 else answer


# while True:
#     try:
#         print(input())
#     except Exception as ec:
#         break


def voca(vocas):
    # "알파벳": 개수
    cnt = {}
    vocas = vocas.upper()

    for v in set(vocas):
        cnt[v] = vocas.count(v)

    flag = max(cnt.values())
    if list(cnt.values()).count(flag) > 1:
        return "?"
    else:
        for alphabet in cnt:
            if cnt[alphabet] == flag:
                return alphabet.upper()


def first_names(names):
    temp = {}
    for idx, name in enumerate(names):
        names[idx] = name[0]

    for n in set(names):
        temp[n] = names.count(n)

    ans = ""
    for key, value in temp.items():
        if value >= 5:
            ans += key

    return "".join(sorted(ans)) if ans else "PREDAJA"


# dp
def dp_card(N, prices):
    dp = [0] * (N + 1)
    prices = [0] + prices

    for i in range(1, N + 1):
        for j in range(1, i + 1):
            if dp[i] < dp[i - j] + prices[j]:
                dp[i] = dp[i - j] + prices[j]
    return dp[N]


# N = int(input())
# p = [0] + list(map(int, input().split()))
# dp = [0] + [sys.maxsize] * N
#
# for i in range(1, N+1):
#     for j in range(1, i+1):
#         dp[i] = min(dp[i], dp[i-j] + p[j])
#
# print(dp[N])

def bee_house(number):
    cnt = 1
    start = 2
    while start <= number:
        start += 6 * cnt
        cnt += 1

    return cnt


def selfDividingNumbers(left, right):
    temp = []
    for i in range(left, right + 1):
        flag = 1
        str_num = str(i)
        if "0" in str_num:
            continue

        for digit in str_num:
            if i % int(digit) > 0:
                flag = 0
                break
        if flag:
            temp.append(i)

    return temp


def mySqrt(x):
    # x = 1인 경우 범위 주의
    for number in range(1, x + 2):
        if number * number > x:
            return number - 1


def vocaCheck(strings):
    cnt = 0

    for string in strings:
        temp = []
        flag = 1

        letter = string[0]
        temp.append(letter)

        for s in string:
            if s == letter:
                continue
            elif s not in temp:
                temp.append(s)
                letter = s
            else:
                flag = 0
                break

        if flag:
            cnt += 1

    return cnt


def goodPwd(passwords):
    vowel = "aeiou"
    for password in passwords:
        n_vowel = 0
        vowel_cnt, consonant_cnt = 0, 0
        temp = ""
        flag = 1
        for p in password:
            # 모음인 경우
            if p in vowel:
                n_vowel += 1
                vowel_cnt += 1
                consonant_cnt *= 0

                if vowel_cnt == 3:
                    flag = 0
                    break
                if temp == p:
                    if p == "e" or p == "o":
                        pass
                    else:
                        flag = 0
                        break
            # 자음인 경우
            else:
                consonant_cnt += 1
                vowel_cnt *= 0

                if consonant_cnt == 3:
                    flag = 0
                    break

                if temp == p:
                    flag = 0
                    break

            temp = p

        if n_vowel == 0:
            flag = 0

        if flag:
            print("<{}> is acceptable.".format(password))
        else:
            print("<{}> is not acceptable.".format(password))


def ox_quiz(results):
    cnt = [0] * len(results)

    for idx in range(len(results)):
        if results[idx] == "O":
            cnt[idx] = 1
            if idx > 0 and cnt[idx - 1] > 0:
                cnt[idx] += cnt[idx - 1]

    return sum(cnt)


temp = []


def stack(i):
    if i[0] == "push":
        temp.append(int(i[1]))
    elif i[0] == "pop":
        if temp:
            print(temp[-1])
            temp.pop()
        else:
            print(-1)
    elif i[0] == "size":
        print(len(temp))
    elif i[0] == "empty":
        print(0 if temp else 1)
    elif i[0] == "top":
        print(temp[-1] if temp else -1)


def game(board, moves):
    # 인덱스 헷갈림
    moves = list(map(lambda x: x - 1, moves))

    temp = []
    cnt = 0
    for move in moves:
        b = 0
        while b < len(board):
            if board[b][move] > 0:
                if temp and temp[-1] == board[b][move]:
                    temp.pop()
                    cnt += 2
                else:
                    temp.append(board[b][move])
                board[b][move] = 0
                break
            b += 1

    return cnt


# 완전 제곱수인지 판별하기
# 순차 탐색시 Time Limit 걸림
# 이진 탐색 O(logn) 이용
def isPerfectSquare(num):
    left, right = 0, num
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == num:
            return True
        elif mid * mid > num:
            right = mid - 1
        else:
            left = mid + 1

    return False


def halvesAreAlike(s: str) -> bool:
    vowel = "aeiou"
    half = len(s) // 2
    s1 = s[:half].lower()
    s2 = s[half:].lower()
    s1_cnt, s2_cnt = 0, 0

    for v in vowel:
        s1_cnt += s1.count(v)
        s2_cnt += s2.count(v)

    return s1_cnt == s2_cnt


# backspace가 맨 앞에 나오는 경우, 그냥 빈칸 처리
def backspaceCompare(s: str, t: str) -> bool:
    s = list(s)
    t = list(t)
    s_cnt = s.count("#")
    while s_cnt > 0:
        s_index = s.index("#")
        if s_index > 0:
            del s[s_index - 1:s_index + 1]
        else:
            del s[0]
        s_cnt -= 1

    t_cnt = t.count("#")
    while t_cnt > 0:
        t_index = t.index("#")
        if t_index > 0:
            del t[t_index - 1:t_index + 1]
        else:
            del t[0]
        t_cnt -= 1

    return s == t


def longestCommonPrefix(strs):
    length = [len(string) for string in strs]
    temp = strs[length.index(min(length))]
    ans = ""

    for idx, ch in enumerate(temp):
        b = 0
        while b < len(strs):
            if ch != strs[b][idx]:
                return ans

            b += 1
        ans += ch

    return ans


def checkRecord(s: str) -> bool:
    if s.count("A") > 1:
        return False

    # 연속된 L
    l = 0
    for record in s:
        if record == "L":
            l += 1
            if l == 3:
                return False
        else:
            l = 0
    return True


def removeOuterParentheses(s: str):
    k = []
    lcnt, rcnt = 0, 0
    temp = ""
    ans = ""

    for ss in s:
        if ss == "(":
            lcnt += 1
        elif ss == ")":
            rcnt += 1
        temp += ss

        if lcnt == rcnt:
            k.append(temp)
            temp = ""
            lcnt = 0
            rcnt = 0

    for kk in k:
        ans += kk[1:-1]

    return ans


def nextGreaterElement(nums1: list[int], nums2: list[int]):
    temp = [-1] * len(nums1)

    for num1 in nums1:
        res = nums2.index(num1)

        for k in nums2[res + 1:]:
            if k > nums2[res]:
                temp[nums1.index(num1)] = k
                break

    return temp


## 점프점프
# def dfs(k):
#     visited[k] = True
#
#     for step in [-bridge[k], bridge[k]]:
#         if 0 <= k + step < N and not visited[k + step]:
#             dfs(k + step)
#
#
# def bfs(k):
#     q = deque([k])
#     visited[k] = True
#
#     while q:
#         step = q.popleft()
#
#         for i in [-bridge[step], bridge[step]]:
#             if 0 <= step + i < N and not visited[step + i]:
#                 q.append(step + i)
#                 visited[step + i] = True
#
#
# N = int(input())
# bridge = list(map(int, input().split()))
# s = int(input()) - 1
#
# visited = [False] * N
# bfs(s)
# print(visited.count(True))


def dp1(n):
    dp = [0] * (n + 1)
    if n <= 2:
        return n

    dp[1] = 1
    dp[2] = 2
    for idx in range(3, n + 1):
        dp[idx] = dp[idx - 1] + dp[idx - 2]

    return dp[n] % 1234567


# T = int(input())
# for _ in range(T):
#     logger = input()
#     front = []
#     back = []
#     # ans = []
#     # point = 0
#     for k in logger:
#         # 시간 초과 발생
#         # if k == "<":
#         #     if point > 0:
#         #         point -= 1
#         # elif k == ">":
#         #     if point < len(ans):
#         #         point += 1
#         # elif k == "-":
#         #     if point > 0:
#         #         ans.pop(point-1)
#         #         point -= 1
#         # else:
#         #     ans.insert(point, k)
#         #     point += 1
#         if k == "<":
#             if front:
#                 back.append(front[-1])
#                 front.pop()
#         elif k == ">":
#             if back:
#                 front.append(back[-1])
#                 back.pop()
#         elif k == "-":
#             if front:
#                 front.pop()
#         else:
#             front.append(k)
#
#     print("".join(front) + "".join(reversed(back)))


def calPoints(ops: list[str]) -> int:
    temp = []
    for op in ops:
        if op == "D":
            temp.append(temp[-1] * 2)
        elif op == "C":
            temp.pop()
        elif op == "+":
            temp.append(temp[-1] + temp[-2])

        else:
            temp.append(int(op))

    return sum(temp)


def convertToTitle(columnNumber: int) -> str:
    ans = ""
    while columnNumber > 0:
        if columnNumber % 26 == 0:
            ans += "Z"
            columnNumber = columnNumber // 26 - 1
        else:
            ans += chr(columnNumber % 26 + 64)
            columnNumber = columnNumber // 26

    return ans[::-1]


# def dfs(x, y):
#     if dp[x][y] > 0:
#         return dp[x][y]
#
#     dp[x][y] = 1
#
#     dx = [-1, 1, 0, 0]
#     dy = [0, 0, -1, 1]
#
#     for i in range(4):
#         nx = x + dx[i]
#         ny = y + dy[i]
#
#         if 0 <= nx < N and 0 <= ny < N and forest[nx][ny] > forest[x][y]:
#             dp[x][y] = max(dp[x][y], dfs(nx, ny) + 1)
#
#     return dp[x][y]

def so(citations: list[int]):
    ans = 0
    for i in range(max(citations) + 1):
        cnt = 0
        for citation in citations:
            if citation - i >= 0:
                cnt += 1

            if cnt >= i:
                ans = max(ans, i)

    return ans


# def check(x, boards):
#     for i in range(x):
#         # 상하 겹침 | 대각 겹침
#         if (boards[i] == boards[x]) or (abs(boards[i] - boards[x]) == abs(i - x)):
#             return False
#
#     return True
#
#
# def n_queen(x, b, n):
#     global cnt
#
#     if x == n:
#         cnt += 1
#
#         return
#     else:
#         for i in range(n):
#             b[x] = i
#             if check(x, b):
#                 n_queen(x+1, b, n)
#
#
# N = int(input())
# board = [15] * N
# cnt = 0
#
# n_queen(0, board, N)
# print(cnt)


def get_com(arr, n):
    res = []
    if n == 0:
        return [[]]

    for i in range(len(arr)):
        elem = arr[i]

        for C in get_com(arr[i + 1:], n - 1):
            res.append([elem] + C)
    return res


def trans(arrs):
    for arr in arrs:
        print(" ".join(list(map(str, arr))))


# # k개중 6개 고름
# while True:
#     ks = list(map(int, input().split()))
#     k = ks[0]
#     if k == 0:
#         break
#
#     s = ks[1:]
#
#     trans(get_com(s, 6))
#     print()

te = [False] * (10000 + 1)
te[0] = te[1] = True

for i in range(2, 101):
    if te[i]:
        continue
    else:
        for j in range(i * 2, 10000 + 1, i):
            if not te[j]:
                te[j] = True


ans = 0
T = int(input())
for _ in range(T):
    N = int(input())

    a = b = N // 2
    for i in range(N // 2):
        if not te[a] and not te[b]:
            print(a, b)
            break
        a -= 1
        b += 1
