"""Speech evaluation on recitation clips from the test split of tarteel-ai/everyayah.

The dataset is pinned to a commit of its Hugging Face repository. Its recordings carry no
documented license, so the clips are for evaluation only: they stay under data/datasets/
(gitignored) and are never committed or redistributed.

Two sets of clips:

  everyayah-clips-20   20 short clips for tests (fetch_clips, load_clips), read from the start of
                       a few shards.
  everyayah-test-1000  the evaluation subset (fetch_subset, load_subset): 1,000 clips of at most
                       30 s, stratified by reciter. configs/evals/everyayah-test-1000.json lists
                       its rows with a hash of each clip's audio and transcript, so a fetch can be
                       checked; the audio and the text themselves are not committed.

A shard holds the audio of a row group in two pages of hundreds of megabytes, and a row can only
be reached by reading its page from the start (kvbound.evals.parquet_pages). The test split is
12 GB, and a uniform sample would read nearly all of it. The subset is therefore drawn in two
stages: the rows within the first POOL_SECONDS of audio of each page are the candidates, and each
reciter's share of the subset is drawn at random among its candidates (select_subset). Rows of a
reciter run in mushaf order, so a reciter is sampled from the passages its pages start with, not
from the whole Quran.

evaluate scores any transcriber, a callable from samples to a Transcript: word and character
error rates against the dataset's text, as it is and after normalize_arabic on both sides, and
real-time factors. kvbound.record defines the metrics (wer, wer_raw, cer, rtf_model, rtf_e2e,
encoder_ms, first_token_ms, decode_ms_per_token). Here they are joined by:

  cer_raw                character error rate of the text as written, as wer_raw is for words.
  generate_ms_per_token  per clip, the time from the end of the encoder pass to the token ids on
                         the host, over the generated tokens (the end-of-sequence token is one):
                         every decoder step, the one over the prompt included.

first_token_ms and decode_ms_per_token need the time of the first token and are absent for a
stack that returns all tokens at once; generate_ms_per_token is there for every stack.

Usage: python -m kvbound.evals.asr fetch
       python -m kvbound.evals.asr subset
       python -m kvbound.evals.asr eval --model whisper-base-ar-quran --backend hf
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from kvbound import audio, record
from kvbound.evals import parquet_pages, text
from kvbound.models import ModelSpec, load_spec, model_entry
from kvbound.nvml_sampler import NvmlSampler
from kvbound.paths import CONFIGS, DATA

DATASET_REPO = "tarteel-ai/everyayah"
DATASET_REVISION = "6ea510862d64f59e555a4a8363eebc0f415621df"
SPLIT = "test"
DATASET_DIR = DATA / "datasets" / "everyayah"
INDEX_FILE = "index.json"

CLIPS = "everyayah-clips-20"
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
CLIPS_DIR = DATASET_DIR / "clips"

SUBSET = "everyayah-test-1000"
SUBSET_CLIPS = 1000
SUBSET_SEED = 0
# One encoder window; longer clips are left out of the subset.
SUBSET_MAX_SECONDS = 30.0
# Seconds of audio from the start of each page whose rows are candidates: about 48 MB of a page.
POOL_SECONDS = 1500.0
# A reciter with fewer candidates than this many times its share gets its next rows too.
POOL_FACTOR = 3
SUBSET_DIR = DATASET_DIR / SUBSET
SUBSET_MANIFEST = CONFIGS / "evals" / f"{SUBSET}.json"

AUDIO_COLUMN = "audio.bytes"
METADATA_COLUMNS = ["duration", "text", "reciter"]
BACKENDS = ("hf", "faster-whisper", "kvbound")
DEVICES = ("cuda", "cpu")


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


@dataclass(frozen=True)
class SubsetClip(Clip):
    """A clip of the evaluation subset, stored as the WAV file the dataset holds."""

    def audio(self) -> np.ndarray:
        return audio.decode_wav((SUBSET_DIR / f"{self.id}.wav").read_bytes())


def clip_id(shard: str, row: int) -> str:
    """ "test00008-0042" for row 42 of test shard 8."""
    return f"test{shard.split('-')[1]}-{row:04d}"


def write_index(directory: Path, clips: Sequence[Clip]) -> None:
    index = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "clips": [asdict(clip) for clip in clips],
    }
    with (directory / INDEX_FILE).open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


def read_index(directory: Path, command: str) -> list[dict[str, Any]]:
    """The clips listed in a directory's index; raises FileNotFoundError when not fetched."""
    path = directory / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m kvbound.evals.asr {command}`")
    return json.loads(path.read_text(encoding="utf-8"))["clips"]


def fetch_clips() -> list[Clip]:
    """Writes CLIPS_PER_SHARD short clips from each of CLIP_SHARDS into CLIPS_DIR.

    Each clip is <id>.npy, decoded to 16 kHz float32; index.json lists them with their source
    row, reciter and transcript.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    clips: list[Clip] = []
    for shard in CLIP_SHARDS:
        path = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/{shard}"
        with fs.open(path, "rb", block_size=1 << 20) as f:
            taken, row = 0, 0
            # Small batches stop the read after the first few megabytes of the audio column.
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4):
                for item in batch.to_pylist():
                    if taken < CLIPS_PER_SHARD and item["duration"] <= MAX_CLIP_SECONDS:
                        clip = Clip(
                            id=clip_id(shard, row),
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
    write_index(CLIPS_DIR, clips)
    return clips


def load_clips() -> list[Clip]:
    """The clips fetch_clips wrote; raises FileNotFoundError when they are not fetched."""
    return [Clip(**clip) for clip in read_index(CLIPS_DIR, "fetch")]


@dataclass(frozen=True)
class Row:
    """A row of the test split: its columns other than the audio, and where its audio lies."""

    shard: str
    row: int
    reciter: str
    duration: float
    text: str
    # Seconds of audio that precede the row in its page: what reading the row costs.
    page_seconds_before: float

    @property
    def page_seconds(self) -> float:
        """Seconds of audio from the start of the row's page to the end of the row."""
        return self.page_seconds_before + self.duration


class HubFile:
    """Ranged reads of one dataset file at the pinned revision, counting the bytes read."""

    def __init__(self, fs: HfFileSystem, file: str) -> None:
        self.fs = fs
        self.path = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/{file}"
        self.bytes_read = 0

    def __call__(self, start: int, end: int) -> bytes:
        data = self.fs.cat_file(self.path, start=start, end=end)
        self.bytes_read += len(data)
        return data

    def parquet(self) -> pq.ParquetFile:
        """The file as pyarrow reads it; small blocks keep footer and column reads small."""
        return pq.ParquetFile(self.fs.open(self.path, "rb", block_size=1 << 18))

    def sha256(self) -> str:
        """The hash the Hub stores for the file; nothing is downloaded."""
        return self.fs.info(self.path)["lfs"]["sha256"]


def split_shards(fs: HfFileSystem) -> list[str]:
    """Repository paths of the test split's shards, in order."""
    root = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/"
    names = [str(name) for name in fs.ls(f"{root}data", detail=False)]
    return sorted(name.removeprefix(root) for name in names if f"/{SPLIT}-" in name)


def audio_pages(file: HubFile, parquet: pq.ParquetFile, group: int) -> dict[str, Any]:
    """Arguments of parquet_pages.read_rows for the audio of one row group."""
    schema = parquet.schema
    column = next(i for i in range(len(schema)) if schema.column(i).path == AUDIO_COLUMN)
    chunk = parquet.metadata.row_group(group).column(column)
    start = chunk.dictionary_page_offset or chunk.data_page_offset
    return {
        "pages": parquet_pages.column_pages(file, start, chunk.total_compressed_size),
        "max_definition_level": schema.column(column).max_definition_level,
        "compression": chunk.compression,
    }


def read_metadata(fs: HfFileSystem) -> list[Row]:
    """Every row of the test split, without its audio: a few megabytes of ranged reads."""
    rows = []
    for shard in split_shards(fs):
        file = HubFile(fs, shard)
        parquet = file.parquet()
        columns = parquet.read(columns=METADATA_COLUMNS).to_pylist()
        first = 0
        for group in range(parquet.metadata.num_row_groups):
            for page in audio_pages(file, parquet, group)["pages"]:
                if page.kind != parquet_pages.DATA_PAGE:
                    continue
                # A dictionary-encoded page stands for the dictionary page its rows index, which
                # holds them in the same order.
                seconds = 0.0
                for row in range(first, first + page.num_values):
                    item = columns[row]
                    rows.append(
                        Row(shard, row, item["reciter"], item["duration"], item["text"], seconds)
                    )
                    seconds += item["duration"]
                first += page.num_values
    return rows


def apportion(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Splits n among the keys in proportion to their counts, by largest remainder.

    Equal remainders go to the key that sorts first.
    """
    whole = sum(counts.values())
    shares = {key: n * count // whole for key, count in counts.items()}
    by_remainder = sorted(counts, key=lambda key: (-(n * counts[key] % whole), key))
    for key in by_remainder[: n - sum(shares.values())]:
        shares[key] += 1
    return shares


def candidate_pool(
    rows: Sequence[Row], share: int, pool_seconds: float, pool_factor: int
) -> list[Row]:
    """The rows of one reciter that its share is drawn from, cheapest to read first.

    They are the rows within the first pool_seconds of their page. When those are fewer than
    pool_factor * share, the reciter's next rows by distance from their page's start are added.
    """
    by_cost = sorted(rows, key=lambda r: (r.page_seconds, r.shard, r.row))
    n_near = sum(row.page_seconds <= pool_seconds for row in by_cost)
    return by_cost[: max(n_near, min(len(by_cost), pool_factor * share))]


def select_subset(
    rows: Sequence[Row],
    n_clips: int = SUBSET_CLIPS,
    seed: int = SUBSET_SEED,
    max_seconds: float = SUBSET_MAX_SECONDS,
    pool_seconds: float = POOL_SECONDS,
    pool_factor: int = POOL_FACTOR,
) -> list[Row]:
    """n_clips rows of at most max_seconds, stratified by reciter, in shard and row order.

    A reciter's share is proportional to its rows of at most max_seconds in the whole split and
    is drawn without replacement from its candidate_pool. Each reciter has its own generator,
    seeded by `seed` and the reciter's name, so its draw does not depend on the other reciters.
    """
    by_reciter: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.duration <= max_seconds:
            by_reciter[row.reciter].append(row)
    shares = apportion({reciter: len(own) for reciter, own in by_reciter.items()}, n_clips)
    chosen = []
    for reciter in sorted(by_reciter):
        pool = candidate_pool(by_reciter[reciter], shares[reciter], pool_seconds, pool_factor)
        chosen += random.Random(f"{seed}/{reciter}").sample(pool, shares[reciter])
    return sorted(chosen, key=lambda r: (r.shard, r.row))


def audio_sha256(samples: np.ndarray) -> str:
    """Hash of decoded audio: the samples as little-endian float32."""
    return hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()


def text_sha256(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def save_clip(row: Row, wav: bytes) -> str:
    """Writes one clip's WAV file into SUBSET_DIR and returns the hash of its audio."""
    name = clip_id(row.shard, row.row)
    samples = audio.decode_wav(wav)
    # The audio must be the row's: its length is the duration column's.
    if abs(len(samples) / audio.SAMPLE_RATE - row.duration) > 1e-3:
        raise ValueError(f"{name}: audio of {len(samples)} samples is not the row's")
    (SUBSET_DIR / f"{name}.wav").write_bytes(wav)
    return audio_sha256(samples)


def download_clips(fs: HfFileSystem, rows: Sequence[Row]) -> tuple[dict[str, str], int]:
    """Writes the WAV file of each row into SUBSET_DIR, and the index.

    Returns the audio hash of each clip by id, and the bytes of audio pages downloaded.
    """
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, dict[int, Row]] = defaultdict(dict)
    for row in rows:
        by_shard[row.shard][row.row] = row
    hashes, downloaded = {}, 0
    for shard, wanted in sorted(by_shard.items()):
        file = HubFile(fs, shard)
        parquet = file.parquet()
        first = 0
        for group in range(parquet.metadata.num_row_groups):
            n_rows = parquet.metadata.row_group(group).num_rows
            # Wanted rows of this group, as indices within it.
            here = {index - first for index in wanted if first <= index < first + n_rows}
            if here:
                pages = audio_pages(file, parquet, group)
                for index, wav in parquet_pages.read_rows(file, rows=here, **pages).items():
                    clip = wanted[first + index]
                    hashes[clip_id(shard, clip.row)] = save_clip(clip, wav)
            first += n_rows
        downloaded += file.bytes_read
        print(f"{shard}: {len(wanted)} clips, {file.bytes_read / 1e6:.0f} MB read", flush=True)
    clips = [
        SubsetClip(clip_id(r.shard, r.row), r.shard, r.row, r.reciter, r.duration, r.text)
        for r in rows
    ]
    write_index(SUBSET_DIR, clips)
    return hashes, downloaded


def build_manifest(
    fs: HfFileSystem, rows: Sequence[Row], chosen: Sequence[Row], hashes: Mapping[str, str]
) -> dict[str, Any]:
    """The committed description of the subset: how it was drawn and what each clip hashes to."""
    eligible = [row for row in rows if row.duration <= SUBSET_MAX_SECONDS]
    by_reciter: dict[str, list[Row]] = defaultdict(list)
    for row in eligible:
        by_reciter[row.reciter].append(row)
    shares = Counter(row.reciter for row in chosen)
    pools = {
        reciter: candidate_pool(own, shares[reciter], POOL_SECONDS, POOL_FACTOR)
        for reciter, own in by_reciter.items()
    }
    return {
        "name": SUBSET,
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "split": SPLIT,
        "selection": {
            "n_clips": SUBSET_CLIPS,
            "seed": SUBSET_SEED,
            "max_seconds": SUBSET_MAX_SECONDS,
            "pool_seconds": POOL_SECONDS,
            "pool_factor": POOL_FACTOR,
        },
        "population": {
            "rows": len(rows),
            "reciters": len({row.reciter for row in rows}),
            "over_max_seconds": len(rows) - len(eligible),
            "eligible": len(eligible),
            "eligible_seconds": round(sum(row.duration for row in eligible), 3),
            "candidates": sum(len(pool) for pool in pools.values()),
        },
        "seconds": round(sum(row.duration for row in chosen), 3),
        "shards": {
            shard: HubFile(fs, shard).sha256() for shard in sorted({r.shard for r in chosen})
        },
        "clips": [
            {
                "id": clip_id(row.shard, row.row),
                "shard": row.shard,
                "row": row.row,
                "reciter": row.reciter,
                "duration": row.duration,
                "audio_sha256": hashes[clip_id(row.shard, row.row)],
                "text_sha256": text_sha256(row.text),
            }
            for row in chosen
        ],
    }


SHA256 = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS = {
    "name": str,
    "dataset": str,
    "revision": str,
    "split": str,
    "selection": dict,
    "population": dict,
    "seconds": float,
    "shards": dict,
    "clips": list,
}
SELECTION_KEYS = {
    "n_clips": int,
    "seed": int,
    "max_seconds": float,
    "pool_seconds": float,
    "pool_factor": int,
}
POPULATION_KEYS = {
    "rows": int,
    "reciters": int,
    "over_max_seconds": int,
    "eligible": int,
    "eligible_seconds": float,
    "candidates": int,
}
CLIP_KEYS = {
    "id": str,
    "shard": str,
    "row": int,
    "reciter": str,
    "duration": float,
    "audio_sha256": str,
    "text_sha256": str,
}


def check_fields(value: Any, keys: Mapping[str, type], where: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{where}: expected the keys {sorted(keys)}")
    for key, kind in keys.items():
        if type(value[key]) is not kind:
            raise ValueError(f"{where}.{key}: expected {kind.__name__}, got {value[key]!r}")


def check_manifest(manifest: Any) -> None:
    """Raises ValueError unless `manifest` is a well-formed description of a subset."""
    check_fields(manifest, MANIFEST_KEYS, "manifest")
    check_fields(manifest["selection"], SELECTION_KEYS, "selection")
    check_fields(manifest["population"], POPULATION_KEYS, "population")
    selection, clips, shards = manifest["selection"], manifest["clips"], manifest["shards"]
    if len(clips) != selection["n_clips"]:
        raise ValueError(f"{len(clips)} clips listed, selection.n_clips is {selection['n_clips']}")
    if not all(SHA256.fullmatch(str(digest)) for digest in shards.values()):
        raise ValueError("shards: each shard needs its sha256")
    for i, clip in enumerate(clips):
        where = f"clips[{i}]"
        check_fields(clip, CLIP_KEYS, where)
        if clip["id"] != clip_id(clip["shard"], clip["row"]) or clip["shard"] not in shards:
            raise ValueError(f"{where}: id {clip['id']} does not name a row of a listed shard")
        if not 0 < clip["duration"] <= selection["max_seconds"]:
            raise ValueError(f"{where}: duration {clip['duration']} is outside the selection")
        if not all(SHA256.fullmatch(clip[key]) for key in ("audio_sha256", "text_sha256")):
            raise ValueError(f"{where}: malformed sha256")
    keys = [(clip["shard"], clip["row"]) for clip in clips]
    if keys != sorted(set(keys)):
        raise ValueError("clips are not unique rows in shard and row order")
    if abs(sum(clip["duration"] for clip in clips) - manifest["seconds"]) > 1e-3:
        raise ValueError("seconds is not the sum of the clips' durations")


def manifest_json(manifest: Mapping[str, Any]) -> str:
    """The manifest as it is committed: one line per clip."""
    head = json.dumps({key: value for key, value in manifest.items() if key != "clips"}, indent=2)
    clips = ",\n".join(f"    {json.dumps(clip)}" for clip in manifest["clips"])
    return f'{head.removesuffix("}").rstrip()},\n  "clips": [\n{clips}\n  ]\n}}\n'


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    check_manifest(manifest)
    return manifest


def fetch_subset(build: bool = False) -> dict[str, Any]:
    """Downloads the evaluation subset into SUBSET_DIR and returns its manifest.

    The clips are those of the committed manifest, and their audio and text must hash to its
    entries. With build=True the subset is drawn afresh and the manifest is written instead.
    """
    fs = HfFileSystem()
    rows = read_metadata(fs)
    if build:
        chosen = select_subset(rows)
    else:
        manifest = load_manifest()
        by_id = {clip_id(row.shard, row.row): row for row in rows}
        chosen = [by_id[clip["id"]] for clip in manifest["clips"]]
    hashes, downloaded = download_clips(fs, chosen)
    print(f"{downloaded / 1e9:.2f} GB of audio pages read for {len(chosen)} clips")
    built = build_manifest(fs, rows, chosen, hashes)
    if build:
        SUBSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with SUBSET_MANIFEST.open("w", encoding="utf-8", newline="\n") as f:
            f.write(manifest_json(built))
    elif built != manifest:
        raise ValueError(f"the fetched clips differ from {SUBSET_MANIFEST}")
    return built


def load_subset() -> list[Clip]:
    """The clips of the evaluation subset, in the manifest's order.

    Raises FileNotFoundError when they are not fetched and ValueError when the fetched clips are
    not the manifest's. The audio is checked against the manifest by verify_subset.
    """
    manifest = load_manifest()
    clips = [SubsetClip(**clip) for clip in read_index(SUBSET_DIR, "subset")]
    listed = [(clip["id"], clip["text_sha256"]) for clip in manifest["clips"]]
    if [(clip.id, text_sha256(clip.text)) for clip in clips] != listed:
        raise ValueError(f"{SUBSET_DIR} does not hold the clips of {SUBSET_MANIFEST}")
    return list(clips)


def verify_subset() -> None:
    """Raises ValueError unless every fetched clip's audio hashes to its manifest entry."""
    digests = {clip["id"]: clip["audio_sha256"] for clip in load_manifest()["clips"]}
    for clip in load_subset():
        if audio_sha256(clip.audio()) != digests[clip.id]:
            raise ValueError(f"{clip.id}: the audio differs from the manifest's")


def spread(clips: Sequence[Clip], n: int) -> list[Clip]:
    """n clips evenly spaced through `clips`, which are in reciter order; all of them for n = 0."""
    if n <= 0 or n >= len(clips):
        return list(clips)
    return [clips[i * len(clips) // n] for i in range(n)]


@dataclass(frozen=True)
class Transcript:
    """One clip through a speech stack.

    Times are wall-clock seconds on the host, with device work synchronized at each boundary.
    """

    text: str
    # Token ids the stack fed its decoder before the first generated one.
    prompt: list[int]
    # Generated ids after the prompt, without the end-of-sequence token.
    tokens: list[int]
    # Samples to text: log-mel front end, model and detokenization.
    e2e_s: float
    # Features on the host to token ids on the host.
    model_s: float
    # The encoder alone.
    encoder_s: float
    # Features on the host to the first generated token; None when the stack does not expose it.
    first_token_s: float | None = None


class Transcriber(Protocol):
    """A speech stack loaded for one model and one configuration.

    engine  the record's engine field: name and version of the stack.
    params  the configuration, in the record's params: backend, device, compute_type, weights,
            kv, beams, whatever else the stack was told, and encoder_ms_covers: what the stack's
            encoder time spans, where that is not kvbound.record's definition of encoder_ms.
    """

    engine: dict[str, Any]
    params: dict[str, Any]

    def __call__(self, samples: np.ndarray) -> Transcript:
        """Transcribes one clip of mono 16 kHz float32 samples, at most 30 s."""
        ...


def load_transcriber(
    backend: str, spec: ModelSpec, device: str, compute_type: str | None, beams: int
) -> Transcriber:
    """The stack `backend` loaded with a fetched model.

    A backend is a module with load(spec, device, compute_type, beams) that returns a
    Transcriber; compute_type None is the backend's default for the device. Each is imported
    only when asked for, because the stacks are optional installs.
    """
    if backend == "hf":
        from kvbound.baselines import hf_whisper  # noqa: PLC0415

        return hf_whisper.load(spec, device, compute_type, beams)
    if backend == "faster-whisper":
        from kvbound.baselines import faster_whisper  # noqa: PLC0415

        return faster_whisper.load(spec, device, compute_type, beams)
    if backend == "kvbound":
        from kvbound import asr  # noqa: PLC0415

        return asr.load(spec, device, compute_type, beams)
    raise ValueError(f"unknown backend {backend!r}; known: {BACKENDS}")


def score(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, text.Edits]:
    """Corpus word and character edits, normalized (wer, cer) and as written (wer_raw, cer_raw)."""
    normalized = [
        [text.normalize_arabic(line) for line in lines] for lines in (references, hypotheses)
    ]
    return {
        "wer": text.wer(*normalized),
        "cer": text.cer(*normalized),
        "wer_raw": text.wer(references, hypotheses),
        "cer_raw": text.cer(references, hypotheses),
    }


def evaluate(
    transcriber: Transcriber, clips: Sequence[Clip], warmup: int = 1
) -> tuple[dict[str, Any], dict[str, Any], list[Transcript]]:
    """Runs every clip through the transcriber, one at a time.

    Returns the record's metrics and per-clip samples, and the transcripts. The first clip runs
    `warmup` times before anything is timed.
    """
    if not clips:
        raise ValueError("no clips to evaluate")
    for _ in range(warmup):
        transcriber(clips[0].audio())
    started = time.perf_counter()
    transcripts = [transcriber(clip.audio()) for clip in clips]
    seconds = time.perf_counter() - started
    prompts = {tuple(t.prompt) for t in transcripts}
    if len(prompts) != 1:
        raise ValueError(f"the stack fed different prompts: {sorted(prompts)}")

    references = [clip.text for clip in clips]
    hypotheses = [t.text for t in transcripts]
    edits = score(references, hypotheses)
    if edits["wer"].reference == 0:
        raise ValueError("the references hold no words")
    audio_seconds = sum(clip.duration for clip in clips)
    # Decoder steps of a clip: its generated tokens and the end-of-sequence token.
    n_tokens = [len(t.tokens) + 1 for t in transcripts]
    samples: dict[str, Any] = {
        "clip": [clip.id for clip in clips],
        "duration_s": [clip.duration for clip in clips],
        "e2e_s": [t.e2e_s for t in transcripts],
        "model_s": [t.model_s for t in transcripts],
        "encoder_s": [t.encoder_s for t in transcripts],
        "n_tokens": n_tokens,
        # Normalized word errors and reference words of each clip.
        "word_errors": [],
        "words": [],
    }
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        words = text.word_edits(text.normalize_arabic(reference), text.normalize_arabic(hypothesis))
        samples["word_errors"].append(words.errors)
        samples["words"].append(words.reference)
    metrics: dict[str, Any] = {name: value.rate for name, value in edits.items()}
    metrics["edits"] = {name: asdict(value) for name, value in edits.items()}
    metrics["rtf_model"] = sum(samples["model_s"]) / audio_seconds
    metrics["rtf_e2e"] = sum(samples["e2e_s"]) / audio_seconds
    metrics["encoder_ms"] = record.summarize([t.encoder_s * 1e3 for t in transcripts])
    first = [t.first_token_s for t in transcripts if t.first_token_s is not None]
    if len(first) == len(transcripts):
        samples["first_token_s"] = first
        metrics["first_token_ms"] = record.summarize([value * 1e3 for value in first])
        # The last token is the end-of-sequence token, on the host when the model is done; a
        # clip with no other token has no interval between tokens.
        metrics["decode_ms_per_token"] = record.summarize(
            [
                (t.model_s - start) * 1e3 / len(t.tokens)
                for t, start in zip(transcripts, first, strict=True)
                if t.tokens
            ]
        )
    metrics["generate_ms_per_token"] = record.summarize(
        [(t.model_s - t.encoder_s) * 1e3 / n for t, n in zip(transcripts, n_tokens, strict=True)]
    )
    metrics["audio_seconds"] = audio_seconds
    metrics["seconds"] = seconds
    return metrics, samples, transcripts


def run(
    model: str,
    backend: str,
    device: str = "cuda",
    compute_type: str | None = None,
    beams: int = 1,
    subset: str = SUBSET,
    max_clips: int = 0,
    warmup: int = 1,
) -> tuple[dict[str, Any], list[Clip], list[Transcript]]:
    """Evaluates one stack on one set of clips.

    Returns the arguments of record.write_record, and the clips with their transcripts.
    """
    spec = load_spec(model)
    if spec.family != "whisper":
        raise ValueError(f"{model} is a {spec.family} model; speech evaluation needs whisper")
    clips = spread(load_subset() if subset == SUBSET else load_clips(), max_clips)
    transcriber = load_transcriber(backend, spec, device, compute_type, beams)
    if device == "cuda":
        with NvmlSampler() as sampler:
            metrics, samples, transcripts = evaluate(transcriber, clips, warmup)
        nvml = sampler.summary()
    else:
        metrics, samples, transcripts = evaluate(transcriber, clips, warmup)
        nvml = None
    result = {
        "kind": "asr",
        "engine": transcriber.engine,
        "model": model_entry(spec),
        "params": {
            **transcriber.params,
            "prompt": transcripts[0].prompt,
            "normalization": "normalize_arabic",
            "subset": subset,
            "n_clips": len(clips),
            "dataset": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "warmup": warmup,
        },
        "metrics": metrics,
        "samples": samples,
        "nvml": nvml,
    }
    return result, clips, transcripts


def write_hypotheses(path: Path, clips: Sequence[Clip], transcripts: Sequence[Transcript]) -> None:
    """Each clip's reference and the stack's text and tokens, for reading the errors."""
    lines = [
        {"id": clip.id, "reference": clip.text, "hypothesis": t.text, "tokens": t.tokens}
        for clip, t in zip(clips, transcripts, strict=True)
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(lines, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="the 20 test clips")
    subset = sub.add_parser("subset", help="the evaluation subset, checked against its manifest")
    subset.add_argument("--build", action="store_true", help="draw it afresh; writes the manifest")
    subset.add_argument("--verify", action="store_true", help="only re-hash the fetched audio")
    run_parser = sub.add_parser("eval", help="error rates and real-time factors of one stack")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--backend", choices=BACKENDS, required=True)
    run_parser.add_argument("--device", choices=DEVICES, default="cuda")
    run_parser.add_argument("--compute-type", help="default: the backend's for the device")
    run_parser.add_argument("--beams", type=int, default=1)
    run_parser.add_argument("--subset", choices=(SUBSET, CLIPS), default=SUBSET)
    run_parser.add_argument("--max-clips", type=int, default=0, help="0: every clip")
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--hypotheses", type=Path, help="write each clip's text to this file")
    run_parser.add_argument("--allow-dirty", action="store_true")
    run_parser.add_argument("--no-record", action="store_true", help="print the result only")
    args = parser.parse_args()

    if args.cmd == "fetch":
        for clip in fetch_clips():
            print(f"{clip.id}  {clip.duration:6.2f} s  {clip.reciter}")
        print(CLIPS_DIR)
    elif args.cmd == "subset":
        if not args.verify:
            manifest = fetch_subset(args.build)
            print(f"{len(manifest['clips'])} clips, {manifest['seconds'] / 3600:.2f} h")
        verify_subset()
        print(SUBSET_DIR)
    else:
        if not args.no_record:
            record.require_clean_tree(args.allow_dirty)
        result, clips, transcripts = run(
            args.model,
            args.backend,
            args.device,
            args.compute_type,
            args.beams,
            args.subset,
            args.max_clips,
            args.warmup,
        )
        m, p = result["metrics"], result["params"]
        print(
            f"{args.model} backend={p['backend']} device={p['device']} "
            f"compute_type={p['compute_type']} beams={p['beams']} prompt={p['prompt']}: "
            f"WER {m['wer']:.4f} (raw {m['wer_raw']:.4f}), CER {m['cer']:.4f} "
            f"(raw {m['cer_raw']:.4f}) on {p['n_clips']} clips of {p['subset']}; "
            f"RTF model {m['rtf_model']:.4f}, end to end {m['rtf_e2e']:.4f}; "
            f"encoder p50 {m['encoder_ms']['p50']:.1f} ms, "
            f"generate p50 {m['generate_ms_per_token']['p50']:.2f} ms/token"
        )
        if args.hypotheses:
            write_hypotheses(args.hypotheses, clips, transcripts)
        if not args.no_record:
            print(record.write_record(**result))


if __name__ == "__main__":
    main()
