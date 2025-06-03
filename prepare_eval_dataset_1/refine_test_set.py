filepath = "test_set.txt"
dest_filepath = "test_set_refined.txt"
with open(filepath, "r", encoding="utf8") as f:
    lines = f.readlines()
    output = []
    for line in lines:
        line = line.strip()
        if line == "" or line.count(".") != 1:
            continue
        output.append(line)
with open(dest_filepath, "w", encoding="utf8") as f:
    for line in output:
        f.write(line)
        f.write("\n")