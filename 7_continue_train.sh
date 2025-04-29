onmt_train -config khmer-viet-no-bpe.yaml -verbose -train_from models/run2/model_step_10000.pt

# import subprocess
# model_name = "target"
# command = ["onmt_train",
#             "-config", "khmer-viet-no-bpe.yaml",
#             "-verbose",
#             "-train_from", "models/run2/model_step_10000.pt"]
# print (" ".join(command))
# # Running the subprocess with the provided command
# # run and capture output
# result = subprocess.run(
#     command,
#     capture_output=True,   # captures both stdout and stderr
#     text=True              # returns strings instead of bytes
# )

# # print return‐code, stdout and stderr
# print("Return code:", result.returncode)
# print("=== STDOUT ===")
# print(result.stdout)
# print("=== STDERR ===")
# print(result.stderr)

