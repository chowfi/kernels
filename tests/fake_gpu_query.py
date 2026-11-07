"""Stand-in for `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`.

Prints the lines of $FAKE_GPU_APPS ("pid, name") whose pid is still alive (not a zombie).
"""
import os
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


for line in Path(os.environ["FAKE_GPU_APPS"]).read_text().splitlines():
    if line.strip() and alive(int(line.split(",")[0])):
        print(line)
