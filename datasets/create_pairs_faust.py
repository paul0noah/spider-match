import random

npairs = 100

pairs = []
for idxX in range(80, 100):
    for idxY in range(80, 100):
        if idxX == idxY:
            continue
        pairs.append(f"tr_reg_0{idxX}-tr_reg_0{idxY}")

pairs = random.sample(pairs, npairs)
with open('pairs_faust.txt', 'a') as the_file:
    for i in range(len(pairs)):
        the_file.write(pairs[i] + '\n')