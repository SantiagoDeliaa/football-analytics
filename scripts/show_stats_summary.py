import json
import sys

stats_file = sys.argv[1] if len(sys.argv) > 1 else 'outputs/optimal_test_1_720p_clip_300-20_stats.json'

with open(stats_file, 'r') as f:
    stats = json.load(f)

# Mostrar solo resumen (sin timeline que es muy grande)
summary = {k: v for k, v in stats.items() if k != 'timeline'}

print(json.dumps(summary, indent=2))
