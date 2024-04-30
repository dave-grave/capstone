with open("epoch.txt") as f:
    for line in f:
        numbers_str = line.split()
        numbers_float = [float(x) for x in numbers_str]
        print(numbers_float[1])