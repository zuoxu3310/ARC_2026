"""Download the original public Walker et al. (2018) behavioral workbook."""
from pathlib import Path
import hashlib
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]
URL="https://sites.socsci.uci.edu/~alns/getSubjData.php"
SHA="199b68795158d5ae494af4a5a4bdc6046d1e7a494faabc074d3d364129690bc5"
DEST=ROOT/"datasets/365_participants_alldata.xls"
if __name__=="__main__":
    DEST.parent.mkdir(parents=True,exist_ok=True)
    payload=DEST.read_bytes() if DEST.exists() else urlopen(URL,timeout=60).read()
    if hashlib.sha256(payload).hexdigest()!=SHA:
        raise SystemExit("Source checksum changed; inspect the release before using it.")
    if not DEST.exists(): DEST.write_bytes(payload)
    print("Verified public workbook:",DEST)
