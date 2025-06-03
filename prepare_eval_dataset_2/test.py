filepath = "test_set_refined_1.txt"
with open (filepath, "r", encoding="utf8") as f:
    lines = f.readlines()
    output = {}
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        else:
            vi, en = line.split(".", 1)
        output[vi.strip()] = en.strip()
source_vi = "source_vi.txt"
groundtruth_en = "groundtruth_en.txt"
with open (source_vi, "w", encoding="utf8") as f:
    with open (groundtruth_en, "w", encoding="utf8") as g:
        for vi, en in output.items():
            f.write(vi)
            f.write("\n")
            g.write(en)
            g.write("\n")