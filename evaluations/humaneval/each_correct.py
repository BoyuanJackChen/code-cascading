
f = open('350M_logs.txt', 'r')
counter = 0
correct = 0

# Read each line in the file
for line in f:
    if line[-1]=="\n":
        index = -4
    else:
        index = -3
    print(line[index])
    if line[index] == '1':
        correct += 1
    counter += 1

# Close the file
print(f"counter is {counter}, correct rate is {correct / counter * 100}%")
f.close()