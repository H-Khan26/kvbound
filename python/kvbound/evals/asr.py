"""Recitation clips for speech tests, from the test split of tarteel-ai/everyayah.

The dataset is pinned to a commit of its Hugging Face repository. Its recordings carry no
documented license, so the clips are for evaluation only: they stay under data/datasets/
(gitignored) and are never committed or redistributed.

fetch_clips reads rows from the start of a few test shards with ranged requests until it has
enough short clips; it does not download a shard. A shard is ordered by reciter, so each of
CLIP_SHARDS gives another reciter. They are the shards whose first rows are short: the audio of
every row before a kept one is read too, about 25 MB in all.

Usage: python -m kvbound.evals.asr fetch
"""

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from kvbound import audio
from kvbound.paths import DATA

DATASET_REPO = "tarteel-ai/everyayah"
DATASET_REVISION = "6ea510862d64f59e555a4a8363eebc0f415621df"
# Test shards the clips come from, a few from the start of each.
CLIP_SHARDS = (
    "data/test-00008-of-00013-9eaed5fb5fe455b0.parquet",
    "data/test-00009-of-00013-cbe112f39e947006.parquet",
    "data/test-00011-of-00013-1a285a00e6fda5f7.parquet",
    "data/test-00012-of-00013-9550ff55e5f66569.parquet",
)
CLIPS_PER_SHARD = 5
# Clips longer than this are skipped: tests run the reference model on the CPU.
MAX_CLIP_SECONDS = 12.0
CLIPS_DIR = DATA / "datasets" / "everyayah" / "clips"
INDEX_FILE = "index.json"


@dataclass(frozen=True)
class Clip:
    id: str
    shard: str
    row: int
    reciter: str
    duration: float
    # Reference transcript, with diacritics.
    text: str

    def audio(self) -> np.ndarray:
        """Mono 16 kHz samples as float32 in [-1, 1)."""
        return np.load(CLIPS_DIR / f"{self.id}.npy", allow_pickle=False)


def fetch_clips() -> list[Clip]:
    """Writes CLIPS_PER_SHARD short clips from each of CLIP_SHARDS into CLIPS_DIR.

    Each clip is <id>.npy, decoded to 16 kHz float32; index.json lists them with their source
    row, reciter and transcript.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    clips: list[Clip] = []
    for shard in CLIP_SHARDS:
        shard_number = shard.split("-")[1]
        path = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/{shard}"
        with fs.open(path, "rb", block_size=1 << 20) as f:
            taken, row = 0, 0
            # Small batches stop the read after the first few megabytes of the audio column.
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4):
                for item in batch.to_pylist():
                    if taken < CLIPS_PER_SHARD and item["duration"] <= MAX_CLIP_SECONDS:
                        clip = Clip(
                            id=f"test{shard_number}-{row:04d}",
                            shard=shard,
                            row=row,
                            reciter=item["reciter"],
                            duration=item["duration"],
                            text=item["text"],
                        )
                        samples = audio.decode_wav(item["audio"]["bytes"])
                        np.save(CLIPS_DIR / f"{clip.id}.npy", samples, allow_pickle=False)
                        clips.append(clip)
                        taken += 1
                    row += 1
                if taken == CLIPS_PER_SHARD:
                    break
    index = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "clips": [asdict(clip) for clip in clips],
    }
    with (CLIPS_DIR / INDEX_FILE).open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return clips


def load_clips() -> list[Clip]:
    """The clips fetch_clips wrote; raises FileNotFoundError when they are not fetched."""
    path = CLIPS_DIR / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m kvbound.evals.asr fetch`")
    index = json.loads(path.read_text(encoding="utf-8"))
    return [Clip(**clip) for clip in index["clips"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_subparsers(dest="cmd", required=True).add_parser("fetch")
    parser.parse_args()
    for clip in fetch_clips():
        print(f"{clip.id}  {clip.duration:6.2f} s  {clip.reciter}")
    print(CLIPS_DIR)


if __name__ == "__main__":
    main()
