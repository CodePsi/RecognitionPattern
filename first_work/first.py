import numpy as np
from PIL import Image
import math
from scipy.stats import kendalltau
from scipy.spatial import distance


def fix(S):
    if type(S[0]) == list:
        return [i for j in S for i in j]
    return S


def abs_distance(S, X, weight: float = 1):
    S = fix(S)
    X = fix(X)
    length = min(len(S), len(X))
    s = 0.0
    for i in range(length):
        s += math.fabs((S[i] - X[i]) * weight)

    return s


def euclid_distance(S, X, weight: float = 1):
    S = fix(S)
    X = fix(X)
    length = min(len(S), len(X))
    s = 0.0
    for i in range(length):
        s += math.pow((S[i] - X[i]) * weight, 2)

    return math.sqrt(s)


def minkowski_distance(S, X, p, weight: float = 1):
    S = fix(S)
    X = fix(X)
    length = min(len(S), len(X))
    s = 0.0
    for i in range(length):
        s += math.pow((S[i] - X[i]) * weight, p)

    return math.pow(s, 1 / p)


def camberro_distance(S, X):
    S = fix(S)
    X = fix(X)
    length = min(len(S), len(X))
    s = 0.0
    for i in range(length):
        s += math.fabs((S[i] - X[i]) / (S[i] + X[i]))

    return s


def dl(xiq, xik):
    if xiq > xik:
        return 1
    elif xiq < xik:
        return -1
    elif xiq == xik:
        return 0


def kendall_distance(S, X):
    n = min(len(S), len(X))
    s = 0.0
    ni = 3
    for i in range(n):
        for j in range(ni):
            for q in range(1, ni - 1):
                for k in range(2, ni):
                    s += math.pow(dl(S[i][q], S[i][k]), i) * math.pow(dl(X[i][q], X[i][k]), j)

    return 1 - (2 / (ni * (ni - 1))) * s


def sgn(v):
    if v > 0:
        return 1
    elif v == 0:
        return 0
    else:
        return -1


def kendall_d2(S, X):
    s = 0.0
    for q in range(0, len(S) - 1):
        for k in range(1, len(S)):
            if q < k:
                s += dl(S[q], S[k]) * dl(X[q], X[k])
    # s = 0.0
    # j = 1
    # for i in range(len(S) - 1):
    #     s += (sgn(S[i] - S[j]) * sgn(X[i] - X[j]))
    #     j += 1

    return 1 - (2 / (len(S) * (len(S) - 1))) * s


def chebyshev_distance(S, X):
    m = math.fabs(S[0] - X[0])
    n = min(len(S), len(X))

    for i in range(n):
        m = max(m, math.fabs(S[i] - X[i]))

    return m


nt = np.array(Image.open('img_0.png').convert('RGB').getdata())
nt1 = np.array(Image.open('img_1.png').convert('RGB').getdata())
# nt1 = np.asarray()
imgs = [Image.open('img_0.png').convert('L'),
        Image.open('img_1.png').convert('L'),
        Image.open('img_2.png').convert('L'),
        Image.open('img_3.png').convert('L'),
        Image.open('img_4.png').convert('L'),
        Image.open('img_5.png').convert('L')]

at = [[1, 2, 3], [2, 3, 4], [1, 2, 7], [4, 2, 1], [1, 1, 1], [2, 4, 0]]
at1 = [[1, 1, 3], [4, 1, 4], [3, 4, 6], [0, 1, 2], [1, 2, 0]]


# 3.1 & 3.2 & 3.3

print('\n\n3.1 & 3.2 & 3.3')

#
# Explanation to 3.3.
# Using Euclid, Minkowski, and Abs methods in above examples shows us, mostly, the same values, and
# in compare with other methods it shows quite high and imprecise values, especially, Abs.
#

for o in range(len(at1) - 1):
    print('\n\n-------------------')

    print('Vals: ', at[o], at[o + 1], '\n\n')

    print('Euclid: ', euclid_distance(at[o], at1[o]))
    print('Euclid (lib): ', distance.euclidean(at[o], at1[o], 1))
    print('Euclid with weight: ', euclid_distance(at[o], at1[o], 2))
    print('Minkowski: ', minkowski_distance(at[o], at1[o], 2, 1))
    print('Minkowski (lib): ', distance.minkowski(at[o], at1[o], 2, 1))
    print('Minkowski with weight: ', minkowski_distance(at[o], at1[o], 2, 1.25))
    print('Abs: : ', abs_distance(at[o], at1[o]))
    print('Abs with weight: ', abs_distance(at[o], at1[o], 1.25))
    print('Camberro: ', camberro_distance(at[o], at1[o]))
    print('Camberro (lib): ', distance.canberra(at[o], at1[o]))
    print('Kendall distance: ', kendall_d2(at[o], at1[o]))
    print('Chebyshev: ', chebyshev_distance(at[o], at1[o]))
    print('Chebyshev (lib): ', distance.chebyshev(at[o], at1[o]))

# 3.4
print('\n\n3.4')
x = [[1, 2, 3, 7, 8, 9], [1, 2, 3, 7, 8, 9], [2, 6, 1, 2, 4, 8]]
x1 = [[1, 2, 3, 7, 8, 9], [9, 8, 7, 3, 2, 1], [0, 2, 3, 2, 4, 9]]

print(kendall_d2(x[0], x1[0]))
print(kendall_d2(x[1], x1[1]))
print(kendall_d2(x[2], x1[2]))

# 3.5
print('\n\n3.5')
x = [[1, 1, 3, 4, 5, 9], [1, 2, 3, 7, 8, 9], [2, 6, 1, 2, 4, 8]]
x1 = [[1, 2, 3, 7, 8, 9], [3, 3, 2, 3, 2, 1], [1, 2, 3, 4, 5, 7]]

print(chebyshev_distance(x[0], x1[2]))

# 3.6
print('\n\n3.6')

x = [[1, 2, 3, 4, 5, 9], [1, 2, 3, 4, 5, 6], [1, 6, 1, 2, 4, 8]]
x1 = [[1, 2, 3, 4, 5, 9], [1, 2, 3, 3, 2, 1], [1, 2, 3, 4, 5, 7]]

print('Chebyshev: ', chebyshev_distance(x[0], x1[0]))
print('Kendall: ', kendall_d2(x[0], x1[0]))

# print(kendall_distance(nt, nt1))
# print(kendall_distance(at, at1))
# for i in range(len(at) - 1):
#     coef, p = kendalltau(at[i], at[i + 1])
#     print(p)
#     print(distance.chebyshev(at[i], at[i + 1]))
# for zx in range(len(at) - 1):
#     print(distance.chebyshev(at[zx], at[zx+1]))
#     print(chebyshev_distance(at[zx], at[zx+1]))
#
# print(euclid_distance(at, at1))


# lab 1.2
print('\n\n----------------Task 1.2----------------\n\n')


def calc_a(Xi, Xj):
    n = len(Xi)
    s = 0
    for k in range(n):
        s += (Xi[k] * Xj[k])

    return s


def calc_b(Xi, Xj):
    n = len(Xi)
    s = 0
    for k in range(n):
        s += (1 - Xi[k]) * (1 - Xj[k])

    return s


def calc_g(Xi, Xj):
    n = len(Xi)
    s = 0
    for k in range(n):
        s += Xi[k] * (1 - Xj[k])

    return s


def calc_h(Xi, Xj):
    n = len(Xi)
    s = 0
    for k in range(n):
        s += (1 - Xi[k]) * Xj[k]

    return s


def russell_rao_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / len(Xi)


def jokard_needman_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / (len(Xi) - calc_b(Xi, Xj))


def dyce_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / (2 * calc_a(Xi, Xj) + calc_g(Xi, Xj) + calc_h(Xi, Xj))


def sokal_sniff_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / (calc_a(Xi, Xj) + 2 * (calc_g(Xi, Xj) + calc_h(Xi, Xj)))


def sokal_mishner_sim_fun(Xi, Xj):
    return (calc_a(Xi, Xj) + calc_b(Xi, Xj)) / len(Xi)


def kulzhinsky_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / (calc_g(Xi, Xj) + calc_h(Xi, Xj))


def yula_sim_fun(Xi, Xj):
    ab = (calc_a(Xi, Xj) * calc_b(Xi, Xj))
    gh = (calc_g(Xi, Xj) * calc_h(Xi, Xj))
    return (ab - gh) / (ab + gh)


def hamming_distance(Xi, Xj):
    n = len(Xi)
    s = 0
    for k in range(n):
        s += abs(Xi[k] - Xj[k])

    return s


fr = [
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 1]
]

# reference
ri = [
    [0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],

]


# 3.1
print('\n\n3.1')

print('Russel and Rao similarity function: ', russell_rao_sim_fun(fr[1], fr[2]))
print('Jokard and Needman similarity function: ', jokard_needman_sim_fun(fr[1], fr[2]))
print('Dyce similarity function: ', dyce_sim_fun(fr[1], fr[2]))
print('Sokal and Sniff similarity function: ', sokal_sniff_sim_fun(fr[1], fr[2]))
print('Sokal and Mishner similarity function: ', sokal_mishner_sim_fun(fr[1], fr[2]))
print('Kulzhinsky similarity function: ', kulzhinsky_sim_fun(fr[1], fr[2]))
print('Yula similarity function: ', yula_sim_fun(fr[1], fr[2]))

# 3.2 & 3.3
print('\n\n3.2 & 3.3')

def my_sim_fun(Xi, Xj):
    return calc_a(Xi, Xj) / (calc_a(Xi, Xj) + (calc_g(Xi, Xj) + calc_h(Xi, Xj)))


for i in range(len(ri)):
    for j in range(len(fr)):
        print('\n-------------')
        print('Reference model: ', ri[i])
        print('Just model: ', fr[j])
        print('Russel and Rao similarity function: ', russell_rao_sim_fun(ri[i], fr[j]))
        print('Jokard and Needman similarity function: ', jokard_needman_sim_fun(ri[i], fr[j]))
        print('Dyce similarity function: ', dyce_sim_fun(ri[i], fr[j]))
        print('Sokal and Sniff similarity function: ', sokal_sniff_sim_fun(ri[i], fr[j]))
        print('Sokal and Mishner similarity function: ', sokal_mishner_sim_fun(ri[i], fr[j]))
        print('Kulzhinsky similarity function: ', kulzhinsky_sim_fun(ri[i], fr[j]))
        print('Yula similarity function: ', yula_sim_fun(ri[i], fr[j]))
        print('My similarity function: ', my_sim_fun(ri[i], fr[j]))

# 3.4
print('\n\n3.4')

tm = [
    [1, 1, 1],
    [1, 0, 0],
]

tm1 = [
    [1, 1, 1],
    [0, 0, 0],
]

print('Russel and Rao similarity function max_value: ', russell_rao_sim_fun(tm[0], tm1[0]))
print('Russel and Rao similarity function min_value: ', russell_rao_sim_fun(tm[1], tm1[1]))

# 3.5
print('\n\n3.5')
hm = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
]

hm1 = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 1],
]

#3.6
print('\n\n3.6')

tm = [
    [1, 1, 1],
    [0, 0, 0],
]

tm1 = [
    [1, 1, 1],
    [0, 0, 0],
]



print(hamming_distance(tm[1], tm1[1]))
print(russell_rao_sim_fun(tm[1], tm1[1]))

# lab 1.3
print('\n\n----------------Task 1.3----------------\n\n')
# 3.1

print('\n\n3.1 & 3.2 & 3.3 & 3.4')


def recognition_by_vector_corner(S, X):
    s = 0.0
    n = min(len(S), len(X))
    for k in range(n):
        s += (S[k] * X[k])

    return math.acos(s / (len(S) * len(X)))


def recognition_by_scalar_product(S, X):
    s = 0.0
    n = min(len(S), len(X))
    for k in range(n):
        s += (S[k] * X[k])

    return s


def recognition_by_belong_to_sphere_area(S, X, R):
    s = 0.0
    n = len(S)
    for k in range(n):
        s += math.pow(X[k] - S[k], 2)

    return math.sqrt(s) <= R


def recognition_by_belong_to_conical_area(S, X, fi):
    su = 0.0
    ss = 0.0
    sx = 0.0
    n = len(S)
    for k in range(n):
        su += (X[k] * S[k])
        ss += math.pow(S[k], 2)
        sx += math.pow(X[k], 2)

    return math.acos(su / (math.sqrt(ss) * math.sqrt(sx))) <= fi


x = [[1, 2, 3, 7, 8, 9], [1, 2, 3, 7, 8, 9], [2, 6, 1, 2, 4, 8]]
x1 = [[1, 2, 3, 7, 8, 9], [9, 8, 7, 3, 2, 1], [0, 2, 3, 2, 4, 9]]
hm = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
]

hm1 = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 1],
]
print(recognition_by_vector_corner(hm[0], hm1[0]))
print(recognition_by_scalar_product(x[0], x1[1]))
print(recognition_by_belong_to_sphere_area(x[0], x1[1], 5))
print(recognition_by_belong_to_conical_area(x[0], x1[1], 5))
