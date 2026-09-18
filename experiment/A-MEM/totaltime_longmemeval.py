import re

log_path = "eval_longmemeval.log"

total_time = 0.0
count = 0

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"memory build done in ([0-9.]+)s", line)
        if m:
            total_time += float(m.group(1))
            count += 1

print("Number of samples:", count)
print("Total time (s):", total_time)
print("Average per sample (s):", total_time / count)
