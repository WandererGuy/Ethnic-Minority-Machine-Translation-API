import os 
file_paths = [
"vocab/example.vocab.src",
"vocab/example.vocab.tgt",
"target.model",
"source.model",
"source.vocab",
"target.vocab"
]
for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)

path = "data/target_source.txt"
with open (path, "r") as f:
    lines = f.readlines()
    for line in lines:
        num_lines = len(lines)
        line = line.strip("\n")
        tab_count = line.count('\t')
        if tab_count != 1:
            raise Exception("each line must have 1 tab only between source and target sentence")

import random 
random.seed(42)
if num_lines > 2000000:
    max_num = 2000000
else:
    max_num = num_lines
# sample 2000000 sentence for train tokenizer 
path = "data/target_source.txt"
with open (path, "r") as f:
    lines = f.readlines()
    source_single_line_ls = []
    target_single_line_ls = []
    for line in lines:
        line = line.strip("\n")
        target, source = line.split("\t")
        source_single_line_ls.append(source)
        target_single_line_ls.append(target)
source_dest_path = "data/source.txt"
new_source_single_line_ls = random.sample(source_single_line_ls, max_num)
new_target_single_line_ls = random.sample(target_single_line_ls, max_num)
with open (source_dest_path, "w") as f:
    for index, line in enumerate(new_source_single_line_ls):
        f.write(line)
        f.write("\n")
    
        
target_dest_path = "data/target.txt"
with open (target_dest_path, "w") as f:
    for index, line in enumerate(new_target_single_line_ls):
        f.write(line)
        f.write("\n")
        
            
