"""Partial reads of a Parquet byte-array column, against pyarrow on a file written here."""

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kvbound.evals import parquet_pages
from kvbound.evals.parquet_pages import DATA_PAGE, DICTIONARY_PAGE, PLAIN, SnappyPrefix


def blob(rng, i):
    """A value of a few kilobytes: noise, a run of one byte, and a repeated phrase."""
    noise = rng.randbytes(rng.randrange(2000, 6000))
    return noise + bytes([i % 256]) * rng.randrange(1, 400) + b"abcdefg" * rng.randrange(0, 60)


@pytest.fixture(scope="module")
def column(tmp_path_factory):
    """A column chunk laid out like a Hub audio shard: the first batch of a row group in one
    dictionary page, the rest in plain pages, inside an optional struct."""
    rng = random.Random(0)
    values = [blob(rng, i) for i in range(100)]
    values[7] = values[3]  # a repeated value has one dictionary entry
    batches = [
        pa.record_batch({"audio": [{"bytes": v, "path": None} for v in part], "n": [0] * len(part)})
        for part in (values[:50], values[50:])
    ]
    path = tmp_path_factory.mktemp("parquet") / "shard.parquet"
    with pq.ParquetWriter(
        path, batches[0].schema, compression="snappy", dictionary_pagesize_limit=4096
    ) as writer:
        writer.write_table(pa.Table.from_batches(batches))
    data = path.read_bytes()
    file = pq.ParquetFile(path)
    chunk = file.metadata.row_group(0).column(0)
    assert chunk.path_in_schema == "audio.bytes"
    return {
        "values": values,
        "data": data,
        "start": chunk.dictionary_page_offset,
        "total": chunk.total_compressed_size,
        "max_definition_level": file.schema.column(0).max_definition_level,
    }


def reader(data, log=None):
    def read(start, end):
        if log is not None:
            log.append((start, end))
        return data[start:end]

    return read


def test_pages_are_a_dictionary_then_its_indices_then_plain_values(column):
    pages = parquet_pages.column_pages(reader(column["data"]), column["start"], column["total"])
    assert [page.kind for page in pages][:2] == [DICTIONARY_PAGE, DATA_PAGE]
    assert pages[0].num_values == 49 and pages[1].num_values == 50
    assert all(page.kind == DATA_PAGE and page.encoding == PLAIN for page in pages[2:])
    assert sum(page.num_values for page in pages[1:]) == len(column["values"])
    assert column["max_definition_level"] == 2


@pytest.mark.parametrize("rows", [[0], [3, 7], [49, 50], [99], [1, 60, 98], list(range(100))])
def test_read_rows_matches_the_written_values(column, rows):
    read = reader(column["data"])
    pages = parquet_pages.column_pages(read, column["start"], column["total"])
    found = parquet_pages.read_rows(
        read, pages, column["max_definition_level"], rows, chunk_bytes=1000
    )
    assert found == {row: column["values"][row] for row in rows}


def test_a_page_is_read_no_further_than_its_last_wanted_row(column):
    log = []
    read = reader(column["data"], log)
    pages = parquet_pages.column_pages(read, column["start"], column["total"])
    del log[:]
    parquet_pages.read_rows(read, pages, column["max_definition_level"], [2], chunk_bytes=1000)
    dictionary = pages[0]
    body = dictionary.offset + dictionary.header_bytes
    # The dictionary is read up to its third entry and the Snappy literal that entry ends in.
    in_dictionary = [(s, e) for s, e in log if body <= s < body + dictionary.compressed_bytes]
    assert max(e for _, e in in_dictionary) - body < dictionary.compressed_bytes // 2
    assert all(e <= pages[2].offset for _, e in log)


def test_read_rows_rejects_rows_outside_the_chunk(column):
    read = reader(column["data"])
    pages = parquet_pages.column_pages(read, column["start"], column["total"])
    with pytest.raises(IndexError):
        parquet_pages.read_rows(read, pages, column["max_definition_level"], [100])


def test_snappy_prefix_handles_split_elements_and_overlapping_copies():
    data = b"ab" * 3000 + bytes(range(256)) * 300 + b"\x00" * 70000
    compressed = pa.compress(data, codec="snappy", asbytes=True)
    assert len(compressed) < len(data) // 4
    for size in (1, 7, 4096):
        stream = SnappyPrefix()
        for start in range(0, len(compressed), size):
            stream.feed(compressed[start : start + size])
            assert data.startswith(bytes(stream.out))
        assert bytes(stream.out) == data
