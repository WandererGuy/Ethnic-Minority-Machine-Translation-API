import os
import uuid
import pandas as pd

# 1. Read your TSV file into a list of dicts
file_path = "/home/manh264/code_linux/NMT_server/src/target_source/ALL_phomt_mtet_deduplicate_vietnamese.txt"
only_viet_rows = []
rows = []
count = 0 
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        count += 1
        print (count)
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            # skip malformed lines
            continue
        en, vi = parts
        rows.append({"English": en, "Vietnamese": vi})
        only_viet_rows.append({"Vietnamese": vi})
# 2. Build a DataFrame and write to Excel
df = pd.DataFrame(rows)
only_viet_df = pd.DataFrame(only_viet_rows)
# you can choose engine="xlsxwriter" or engine="openpyxl" if you like:
df.to_csv("en_vi_label.csv", index=False, encoding="utf-8")
only_viet_df.to_csv("only_vi_label.csv", index=False, encoding="utf-8")


# import pandas as pd

# # 1) Your input CSV
# csv_path = "only_vi_label.csv"

# # 2) Choose a chunk‐size smaller than Excel’s row limit.
# #    Excel max is 1 048 576 rows per sheet, so we’ll use 1 000 000.
# chunksize = 1_000_000

# # 3) Iterate over the CSV in chunks…
# file_counter = 1
# for chunk in pd.read_csv(csv_path, chunksize=chunksize, iterator=True, encoding="utf-8"):
#     # 4) Write each chunk to a new Excel file
#     out_xlsx = f"output_part_{file_counter:03d}.xlsx"
#     chunk.to_excel(out_xlsx, index=False, engine="xlsxwriter")
#     print(f"Wrote {len(chunk)} rows to {out_xlsx}")
#     file_counter += 1
