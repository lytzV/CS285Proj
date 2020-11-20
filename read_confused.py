import os

confused = {}
f = open('confusion.txt',"r")
for line in f:
    key = line[0]
    confusions = [s for s in line[2:]][:-1]
    confused[key] = confusions

print(confused['它'])
print(confused['汤'])
print(confused['铟'])
print(confused['改'])
print(confused['捐'])
print(confused['迈'])
print(confused['祢'])
print(confused['单'])
print(confused['仨'])