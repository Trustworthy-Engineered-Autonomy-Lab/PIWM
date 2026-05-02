import re

clean = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        # Drop conda-internal paths
        if "@ file:///" in line:
            continue
        if "conda" in line.lower():
            continue
        if "mamba" in line.lower():
            continue
        if line == "":
            continue
        clean.append(line)

with open("requirements_cleaned.txt", "w") as f:
    f.write("\n".join(clean))

print("Cleaned file written to requirements_cleaned.txt")
