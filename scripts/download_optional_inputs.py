"""Download optional public inputs and verify their archived SHA-256 checksums."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['atlases', 'participant-list'])
    args = parser.parse_args()
    records = json.loads((ROOT / 'docs/optional_input_manifest.json').read_text())
    for record in records:
        if record['group'] != args.group:
            continue
        path = ROOT / record['destination']
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
            print('Verified existing', record['destination'])
            continue
        request = urllib.request.Request(record['url'], headers={'User-Agent': 'ARC_2026 reproduction'})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        assert hashlib.sha256(content).hexdigest() == record['sha256'], record['url']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print('Downloaded and verified', record['destination'])
