
counter = 0
file = open("09_results.txt", "r")
for line in file:
    if line.startswith("result is ['failed: invalid syntax "):
        counter += 1
    elif line.startswith("result is ['failed: unindent "):
        counter += 1
    elif line.startswith("result is ['failed: name "):
        counter += 1
    elif line.startswith("result is ['failed: syntax"):
        counter += 1
file.close()
print(counter/164)