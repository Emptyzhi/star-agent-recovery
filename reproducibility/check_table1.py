"""Recompute every main-table result directly from released task records."""
import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'reproducibility/table1_index.json').read_text())
    observed = defaultdict(dict)
    total = 0
    for entry in manifest['records']:
        rows = list(csv.DictReader((ROOT / entry['tasks_file']).open(encoding='utf-8-sig')))
        size = 134 if entry['dataset'] == 'ALFWorld' else 500
        assert len(rows) == size == entry['tasks'], entry
        assert sorted(int(row['task_id']) for row in rows) == list(range(size)), entry
        assert all(row['success'] in ['0', '1', '0.0', '1.0'] for row in rows), entry
        successes = sum(int(float(row['success'])) for row in rows)
        assert successes == entry['successes'], entry
        sr = 100 * successes / size
        reward = 100 * sum(float(row['reward']) for row in rows) / size
        assert abs(sr - entry['SR_percent']) < 1e-8, entry
        assert abs(reward - entry['R_percent']) < 1e-8, entry
        observed[(entry['model'], entry['method'])][entry['dataset']] = (sr, reward)
        total += size
    for expected in manifest['expected_table_rows']:
        values = observed[(expected['model'], expected['method'])]
        computed = dict(Game24_SR=values['Game24'][0], ALFWorld_SR=values['ALFWorld'][0],
                        WebShop_R=values['WebShop'][1], WebShop_SR=values['WebShop'][0],
                        Avg_SR=sum(values[ds][0] for ds in ['Game24','ALFWorld','WebShop'])/3)
        for key, value in computed.items():
            assert f'{value:.1f}' == f'{expected[key]:.1f}', (expected, key, value)
        print(expected['model'], expected['method'], ' '.join(f'{v:.1f}' for v in computed.values()))
    print(f'PASS: {len(manifest["records"])} cohorts, {total} task records, all 21 main-table rows.')


if __name__ == '__main__':
    main()
