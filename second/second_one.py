input = [
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0,
     1, 0, 0, 1, 0,
     1, 1, 1, 0, 0,
     1, 0, 0, 1, 0,
     1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1]
]

out = [1, 1, 1, 1]

train_set = [
    [1, 0, 1, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1]
]


def output_signal(s):
    if s > 0:
        return 1
    else:
        return 0


def dl_w(xi, y):
    if xi * y == 1:
        return 1
    elif xi == 0:
        return 0
    elif xi != 0 and y == 0:
        return -1


weights = [0 for _ in range(len(input[0]))]


def train(input_args, satisfiable_output_args):
    for inputs in range(len(input_args)):
        inp = input_args[inputs]
        x0 = inp[0]
        y = satisfiable_output_args[inputs]
        stop = True
        s = 0
        while stop:
            for i in range(len(weights)):
                weights[i] = weights[i] + dl_w(inp[i], y)

            for i in range(len(inp)):
                s += inp[i] * weights[i] + weights[0]

            probable = output_signal(s)
            stop = probable != y

        return True


print(train(input, out))
print(weights)

# [0, 0, 1, 0, 0,
#  0, 1, 0, 1, 0,
#  1, 1, 1, 1, 1,
#  1, 0, 0, 0, 1,
#  1, 0, 0, 0, 1]
# print(train(train_set))