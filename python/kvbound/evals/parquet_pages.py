"""Reads chosen rows of a remote Parquet byte-array column without downloading its pages whole.

Audio datasets on the Hugging Face Hub written by old `datasets` versions hold the audio of a
1,000-row group in two pages of hundreds of megabytes: a dictionary page with the first rows
(Arrow notices that the dictionary outgrew its limit only after a whole batch went in) and a
plain data page with the rest. pyarrow reads a page whole. A page is one raw Snappy stream with
no index, so a value can only be reached by decompressing from the start of its page, but the
stream need not be read past the value. read_rows does that: it walks the page headers with
small ranged reads, then downloads each page only as far as the last row asked for.

Supported: a BYTE_ARRAY column that is not repeated and holds no nulls, Snappy compression,
version 1 data pages, plain and dictionary encodings.
"""

import struct
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

# Bytes [start, end) of the file.
Read = Callable[[int, int], bytes]

DATA_PAGE = 0
DICTIONARY_PAGE = 2
PLAIN = 0
DICTIONARY_ENCODINGS = (2, 8)  # PLAIN_DICTIONARY, RLE_DICTIONARY
# First guess at the size of a page header; a header that holds statistics can be longer.
HEADER_BYTES = 1024
CHUNK_BYTES = 4 << 20


@dataclass(frozen=True)
class Page:
    # File offset of the page header, and the header's length.
    offset: int
    header_bytes: int
    # DATA_PAGE or DICTIONARY_PAGE.
    kind: int
    compressed_bytes: int
    uncompressed_bytes: int
    num_values: int
    encoding: int


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _thrift_value(buf: bytes, pos: int, kind: int) -> tuple[Any, int]:
    """One value of the Thrift compact protocol; doubles are skipped."""
    if kind in (1, 2):  # true, false
        return kind == 1, pos
    if kind == 3:  # byte
        return buf[pos], pos + 1
    if kind in (4, 5, 6):  # i16, i32, i64
        n, pos = _varint(buf, pos)
        return _zigzag(n), pos
    if kind == 7:  # double
        return None, pos + 8
    if kind == 8:  # binary
        n, pos = _varint(buf, pos)
        if pos + n > len(buf):
            raise IndexError("binary runs past the buffer")
        return buf[pos : pos + n], pos + n
    if kind in (9, 10):  # list, set
        head = buf[pos]
        pos += 1
        size, element = head >> 4, head & 15
        if size == 15:
            size, pos = _varint(buf, pos)
        items = []
        for _ in range(size):
            item, pos = _thrift_value(buf, pos, element)
            items.append(item)
        return items, pos
    if kind == 12:
        return _thrift_struct(buf, pos)
    raise ValueError(f"unsupported Thrift type {kind}")


def _thrift_struct(buf: bytes, pos: int) -> tuple[dict[int, Any], int]:
    """A Thrift compact struct as {field id: value}, and the position after it.

    Raises IndexError when the struct runs past the buffer.
    """
    fields: dict[int, Any] = {}
    field = 0
    while True:
        head = buf[pos]
        pos += 1
        if head == 0:
            return fields, pos
        delta, kind = head >> 4, head & 15
        if delta:
            field += delta
        else:
            n, pos = _varint(buf, pos)
            field = _zigzag(n)
        fields[field], pos = _thrift_value(buf, pos, kind)


def column_pages(read: Read, start: int, total_bytes: int) -> list[Page]:
    """The pages of a column chunk that starts at file offset `start` and spans `total_bytes`.

    Reads the page headers only: one small request per page.
    """
    pages = []
    pos, end = start, start + total_bytes
    while pos < end:
        size = HEADER_BYTES
        while True:
            try:
                header, used = _thrift_struct(read(pos, min(pos + size, end)), 0)
                break
            except IndexError:
                if pos + size >= end:
                    raise ValueError(f"page header at {pos} runs past the column chunk") from None
                size *= 4
        # PageHeader: 1 type, 2 uncompressed size, 3 compressed size, 5 data page header,
        # 7 dictionary page header; both sub-headers hold 1 num_values and 2 encoding.
        kind = header[1]
        detail = header.get(7 if kind == DICTIONARY_PAGE else 5)
        if detail is None:
            raise ValueError(f"page type {kind} at {pos} is not supported")
        pages.append(Page(pos, used, kind, header[3], header[2], detail[1], detail[2]))
        pos += used + header[3]
    if pos != end:
        raise ValueError(f"pages end at {pos}, the column chunk at {end}")
    return pages


class SnappyPrefix:
    """Decompresses a raw Snappy stream from its start, as far as its bytes have been fed.

    `out` holds everything decompressed so far; back-references read from it.
    """

    def __init__(self) -> None:
        self.out = bytearray()
        self._pending = b""
        self._length: int | None = None

    def feed(self, data: bytes) -> None:
        buf = self._pending + data
        pos, out = 0, self.out
        if self._length is None:
            # The stream opens with its decompressed length.
            if not any(byte < 0x80 for byte in buf[:5]):
                self._pending = buf
                return
            self._length, pos = _varint(buf, 0)
        while pos < len(buf):
            tag = buf[pos]
            if tag & 3 == 0:
                # Literal: lengths above 60 follow the tag in 1 to 4 bytes.
                extra = max(0, (tag >> 2) - 59)
                body = pos + 1 + extra
                if body > len(buf):
                    break
                n = 1 + (int.from_bytes(buf[pos + 1 : body], "little") if extra else tag >> 2)
                if body + n > len(buf):
                    break
                out += buf[body : body + n]
                pos = body + n
                continue
            # Copy of `n` bytes that start `offset` bytes back in the output.
            width = (2, 3, 5)[(tag & 3) - 1]
            if pos + width > len(buf):
                break
            if tag & 3 == 1:
                n, offset = 4 + ((tag >> 2) & 7), (tag >> 5) << 8 | buf[pos + 1]
            else:
                n, offset = 1 + (tag >> 2), int.from_bytes(buf[pos + 1 : pos + width], "little")
            pos += width
            if not 0 < offset <= len(out):
                raise ValueError("corrupt Snappy stream: copy from before the start")
            source = len(out) - offset
            if offset >= n:
                out += out[source : source + n]
            else:
                # The copy overlaps its own output: the last `offset` bytes repeat.
                out += (out[source:] * (n // offset + 1))[:n]
        self._pending = buf[pos:]
        if len(out) > self._length:
            raise ValueError("corrupt Snappy stream: longer than its declared length")


def _rle_hybrid(buf: bytes, bit_width: int, count: int) -> list[int]:
    """`count` values of Parquet's RLE / bit-packed hybrid encoding."""
    values: list[int] = []
    pos, mask = 0, (1 << bit_width) - 1
    while len(values) < count:
        header, pos = _varint(buf, pos)
        if header & 1:
            # Groups of 8 values, bit_width bits each, least significant bit first.
            n_bytes = (header >> 1) * bit_width
            bits = int.from_bytes(buf[pos : pos + n_bytes], "little")
            values += [(bits >> (i * bit_width)) & mask for i in range((header >> 1) * 8)]
            pos += n_bytes
        else:
            n_bytes = (bit_width + 7) // 8
            values += [int.from_bytes(buf[pos : pos + n_bytes], "little")] * (header >> 1)
            pos += n_bytes
    return values[:count]


class _PagePrefix:
    """The decompressed start of a data or dictionary page, downloaded as it is asked for."""

    def __init__(self, read: Read, page: Page, chunk_bytes: int) -> None:
        self._read, self._chunk_bytes = read, chunk_bytes
        self._next = page.offset + page.header_bytes
        self._end = self._next + page.compressed_bytes
        self._stream = SnappyPrefix()

    def take(self, start: int, n: int) -> bytes:
        """Decompressed bytes [start, start + n) of the page."""
        while len(self._stream.out) < start + n:
            if self._next >= self._end:
                raise ValueError("the page ends inside a value")
            end = min(self._next + self._chunk_bytes, self._end)
            self._stream.feed(self._read(self._next, end))
            self._next = end
        return bytes(self._stream.out[start : start + n])


def _skip_levels(prefix: _PagePrefix, page: Page, max_definition_level: int) -> int:
    """The position after a data page's definition levels; raises if a value is null."""
    if max_definition_level == 0:
        return 0
    (n,) = struct.unpack("<I", prefix.take(0, 4))
    width = max_definition_level.bit_length()
    levels = _rle_hybrid(prefix.take(4, n), width, page.num_values)
    if any(level != max_definition_level for level in levels):
        raise ValueError("the column holds nulls")
    return 4 + n


def _plain_values(prefix: _PagePrefix, pos: int, wanted: Collection[int]) -> dict[int, bytes]:
    """The PLAIN byte-array values with the indices `wanted`, the first value being at `pos`.

    A value is a 4-byte length, then the bytes. The page is downloaded up to the last one wanted.
    """
    values = {}
    for i in range(max(wanted) + 1):
        (n,) = struct.unpack("<I", prefix.take(pos, 4))
        if i in wanted:
            values[i] = prefix.take(pos + 4, n)
        pos += 4 + n
    return values


def read_rows(
    read: Read,
    pages: Sequence[Page],
    max_definition_level: int,
    rows: Collection[int],
    compression: str = "SNAPPY",
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[int, bytes]:
    """The values of `rows` (indices within the row group) of a column chunk with `pages`.

    max_definition_level is the column's, from the file schema. Every page that holds one of
    the rows is downloaded up to the last of them it holds.
    """
    if compression != "SNAPPY":
        raise ValueError(f"compression {compression} is not supported")
    wanted = set(rows)
    found: dict[int, bytes] = {}
    dictionary: Page | None = None
    # Dictionary index of each wanted row that a dictionary-encoded page holds.
    indices: dict[int, int] = {}
    first = 0
    for page in pages:
        if page.kind == DICTIONARY_PAGE:
            dictionary = page
            continue
        # Wanted rows of this page, as indices within it.
        here = {row - first for row in wanted if first <= row < first + page.num_values}
        first += page.num_values
        if not here:
            continue
        prefix = _PagePrefix(read, page, chunk_bytes)
        pos = _skip_levels(prefix, page, max_definition_level)
        if page.encoding in DICTIONARY_ENCODINGS:
            # One byte of bit width, then the indices; such a page is small and read whole.
            body = prefix.take(pos, page.uncompressed_bytes - pos)
            page_indices = _rle_hybrid(body[1:], body[0], page.num_values)
            indices.update({first - page.num_values + i: page_indices[i] for i in here})
        elif page.encoding == PLAIN:
            values = _plain_values(prefix, pos, here)
            found.update({first - page.num_values + i: value for i, value in values.items()})
        else:
            raise ValueError(f"page encoding {page.encoding} is not supported")
    if indices:
        if dictionary is None or dictionary.encoding not in (PLAIN, *DICTIONARY_ENCODINGS):
            raise ValueError("dictionary-encoded pages without a plain dictionary page")
        if max(indices.values()) >= dictionary.num_values:
            raise ValueError("a dictionary index is outside the dictionary")
        prefix = _PagePrefix(read, dictionary, chunk_bytes)
        entries = _plain_values(prefix, 0, set(indices.values()))
        found.update({row: entries[index] for row, index in indices.items()})
    if missing := wanted - set(found):
        raise IndexError(f"rows {sorted(missing)} are outside the column chunk")
    return found
