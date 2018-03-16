import sys
import pickle
sys.path.append("../JumpToMachineLearning/AppData/")

# need it one time for python 3
original = "../JumpToMachineLearning/AppData/email_authors.pkl"
destination = "../JumpToMachineLearning/AppData/email_authors.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))


original = "../JumpToMachineLearning/AppData/word_data.pkl"
destination = "../JumpToMachineLearning/AppData/word_data.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

