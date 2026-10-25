"""Recompute STAR score and available cost summaries from released task records."""
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'runs.json').read_text(encoding='utf-8'))
    summaries = []
    for record in manifest['records']:
        with (ROOT / record['tasks_file']).open(encoding='utf-8', newline='') as handle:
            tasks = list(csv.DictReader(handle))
        expected = 134 if record['dataset'] == 'ALFWorld' else 500
        if sorted(int(row['task_id']) for row in tasks) != list(range(expected)):
            raise ValueError('Missing or duplicated task IDs: ' + record['record'])
        successes = sum(int(row['success']) for row in tasks)
        sr = 100 * successes / expected
        reward = 100 * math.fsum(float(row['reward']) for row in tasks) / expected
        if successes != record['successes'] or not math.isclose(sr, record['SR_percent']):
            raise ValueError('Success count differs from the source record')
        if not math.isclose(reward, record['R_percent']):
            raise ValueError('Reward differs from the source record')
        result = {'record': record['record'], 'tasks': expected, 'SR_percent': sr,
                  'reward_percent': reward}
        for field, reference in record.get('cost_means', {}).items():
            values = [float(row[field]) for row in tasks]
            mean = math.fsum(values) / expected
            if not math.isclose(mean, reference):
                raise ValueError('Cost differs from the source record')
            result[field] = mean
        summaries.append(result)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
