#!/usr/bin/env python3
"""Print the sweap-images tags for a slice of the dataset (one per line)."""
import json, sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for i, line in enumerate(open(path)):
    if start <= i < end:
        print("jefzda/sweap-images:" + json.loads(line)["dockerhub_tag"])
