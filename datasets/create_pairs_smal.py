import random
import os

npairs = 100
pairs = []
files = []
with open(os.path.join('SMAL_r', 'test.txt')) as testset:
    for line in testset:
        files.append(line.strip() + '.off')
for idxX in range(0, len(files)):
    for idxY in range(idxX+1, len(files)):
        if idxX == idxY:
            continue
        pairs.append(files[idxX][:-4] + '-' + files[idxY][:-4])

pairs = random.sample(pairs, npairs)
with open('pairs_smal.txt', 'a') as the_file:
    for i in range(len(pairs)):
        the_file.write(pairs[i] + '\n')