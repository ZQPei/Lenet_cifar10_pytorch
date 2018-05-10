import os

params = {}

with open('/home/pzq/project/pt_cifar10/src/params.txt','r') as fo:
    lines = fo.readlines()
    for eachline in lines:
        line = eachline.strip()
        if line == '' or line[0] == '#':
            continue
        key, value = line.split('=')
        print(key,value)
        params[key] = value

#print(params)

if __name__ == "__main__":
    pass