import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

CONSOLE = Console()
SNAPPY_CODEC = pa.Codec("snappy")

SCHEMA = pa.schema([
    ("source", pa.string()),
    ("data_id", pa.string()),
    ("diffusion_itr_id", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("answer_trajectory_ids", pa.list_(pa.list_(pa.int32()))),
    ("teacher_output_ids", pa.list_(pa.int32())),
])

TSTOP = 0
TTRUE = 1
TFALSE = 2
TBYTE = 3
TI16 = 4
TI32 = 5
TI64 = 6
TDOUBLE = 7
TBINARY = 8
TLIST = 9
TSET = 10
TMAP = 11
TSTRUCT = 12

PAGE_TYPE_DATA_PAGE = 0
PAGE_TYPE_DICTIONARY_PAGE = 2
ENCODING_PLAIN = 0
ENCODING_PLAIN_DICTIONARY = 2
ENCODING_RLE_DICTIONARY = 8


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    max_rep: int
    max_def: int
    physical_type: str
    builder: str


@dataclass
class PageHeader:
    page_type: int | None
    uncompressed_size: int | None
    compressed_size: int | None
    data_page_header: dict[int, object] | None
    dictionary_page_header: dict[int, object] | None


@dataclass
class ColumnChunk:
    next_pos: int
    row_count: int
    rep_levels: list[int]
    def_levels: list[int]
    values: list[object]


COLUMN_SPECS = [
    ColumnSpec("source", 0, 1, "BYTE_ARRAY", "scalar"),
    ColumnSpec("data_id", 0, 1, "BYTE_ARRAY", "scalar"),
    ColumnSpec("diffusion_itr_id", 0, 1, "BYTE_ARRAY", "scalar"),
    ColumnSpec("prompt_ids", 1, 3, "INT32", "list"),
    ColumnSpec("answer_trajectory_ids", 2, 5, "INT32", "nested_list"),
    ColumnSpec("teacher_output_ids", 1, 3, "INT32", "list"),
]


def read_u32_le(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def read_i32_le(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<i", data, pos)[0], pos + 4


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def read_compact_int(data: bytes, pos: int) -> tuple[int, int]:
    value, pos = read_varint(data, pos)
    return zigzag_decode(value), pos


def read_compact_binary(data: bytes, pos: int) -> tuple[bytes, int]:
    size, pos = read_varint(data, pos)
    return data[pos:pos + size], pos + size


def read_compact_value(data: bytes, pos: int, field_type: int):
    if field_type == TTRUE:
        return True, pos
    if field_type == TFALSE:
        return False, pos
    if field_type == TBYTE:
        return struct.unpack_from("<b", data, pos)[0], pos + 1
    if field_type in (TI16, TI32, TI64):
        return read_compact_int(data, pos)
    if field_type == TDOUBLE:
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if field_type == TBINARY:
        return read_compact_binary(data, pos)
    if field_type == TSTRUCT:
        return read_compact_struct(data, pos)
    if field_type in (TLIST, TSET):
        header = data[pos]
        pos += 1
        size = header >> 4
        elem_type = header & 0x0F
        if size == 15:
            size, pos = read_varint(data, pos)
        items = []
        for _ in range(size):
            item, pos = read_compact_value(data, pos, elem_type)
            items.append(item)
        return items, pos
    if field_type == TMAP:
        size, pos = read_varint(data, pos)
        if size == 0:
            return {}, pos
        key_value_type = data[pos]
        pos += 1
        key_type = key_value_type >> 4
        value_type = key_value_type & 0x0F
        out = {}
        for _ in range(size):
            key, pos = read_compact_value(data, pos, key_type)
            value, pos = read_compact_value(data, pos, value_type)
            out[key] = value
        return out, pos
    raise ValueError(f"Unsupported compact field type: {field_type}")


def read_compact_struct(data: bytes, pos: int) -> tuple[dict[int, object], int]:
    result: dict[int, object] = {}
    last_field_id = 0
    while True:
        header = data[pos]
        pos += 1
        field_type = header & 0x0F
        if field_type == TSTOP:
            return result, pos
        delta = header >> 4
        if delta == 0:
            field_id, pos = read_compact_int(data, pos)
        else:
            field_id = last_field_id + delta
        value, pos = read_compact_value(data, pos, field_type)
        result[field_id] = value
        last_field_id = field_id


def parse_page_header(data: bytes, pos: int) -> tuple[PageHeader, int]:
    raw, pos = read_compact_struct(data, pos)
    return PageHeader(
        page_type=raw.get(1),
        uncompressed_size=raw.get(2),
        compressed_size=raw.get(3),
        data_page_header=raw.get(5),
        dictionary_page_header=raw.get(7),
    ), pos


def decode_rle_bitpacked(data: bytes, bit_width: int, count: int) -> list[int]:
    values: list[int] = []
    pos = 0
    mask = (1 << bit_width) - 1 if bit_width else 0

    while len(values) < count:
        header, pos = read_varint(data, pos)
        if header & 1 == 0:
            run_length = header >> 1
            byte_width = (bit_width + 7) // 8
            raw = int.from_bytes(data[pos:pos + byte_width], "little") if byte_width else 0
            pos += byte_width
            values.extend([raw] * run_length)
            continue

        num_groups = header >> 1
        total = num_groups * 8
        byte_count = (total * bit_width + 7) // 8
        chunk = data[pos:pos + byte_count]
        pos += byte_count

        bit_pos = 0
        for _ in range(total):
            value = 0
            for bit_idx in range(bit_width):
                byte_idx = (bit_pos + bit_idx) // 8
                shift = (bit_pos + bit_idx) % 8
                value |= ((chunk[byte_idx] >> shift) & 1) << bit_idx
            values.append(value & mask)
            bit_pos += bit_width

    return values[:count]


def decode_levels(data: bytes, max_level: int, count: int) -> tuple[list[int], int]:
    if count == 0:
        return [], 0
    if max_level == 0:
        return [0] * count, 0

    byte_len, pos = read_u32_le(data, 0)
    levels = decode_rle_bitpacked(data[pos:pos + byte_len], max_level.bit_length(), count)
    return levels, 4 + byte_len


def decode_dictionary_page(payload: bytes, header: PageHeader, physical_type: str) -> list[object]:
    if header.dictionary_page_header is None:
        raise ValueError("Missing dictionary page header")

    num_values = header.dictionary_page_header[1]
    encoding = header.dictionary_page_header[2]
    if encoding != ENCODING_PLAIN:
        raise ValueError(f"Unsupported dictionary encoding: {encoding}")

    values = []
    pos = 0
    for _ in range(num_values):
        if physical_type == "BYTE_ARRAY":
            size, pos = read_i32_le(payload, pos)
            raw = payload[pos:pos + size]
            pos += size
            values.append(raw.decode("utf-8"))
        elif physical_type == "INT32":
            value, pos = read_i32_le(payload, pos)
            values.append(value)
        else:
            raise ValueError(f"Unsupported physical type: {physical_type}")
    return values


def decode_dictionary_indices(payload: bytes, count: int) -> list[int]:
    return decode_rle_bitpacked(payload[1:], payload[0], count)


def decode_data_page(payload: bytes, header: PageHeader, max_rep: int, max_def: int, dictionary: list[object] | None):
    if header.data_page_header is None:
        raise ValueError("Missing data page header")

    num_values = header.data_page_header[1]
    encoding = header.data_page_header[2]

    rep_levels, offset = decode_levels(payload, max_rep, num_values)
    def_levels, consumed = decode_levels(payload[offset:], max_def, num_values)
    offset += consumed

    if encoding not in (ENCODING_RLE_DICTIONARY, ENCODING_PLAIN_DICTIONARY):
        raise ValueError(f"Unsupported data page encoding: {encoding}")
    if dictionary is None:
        raise ValueError("Dictionary page is required before data page")

    value_count = sum(level == max_def for level in def_levels)
    values = [dictionary[idx] for idx in decode_dictionary_indices(payload[offset:], value_count)]
    return rep_levels, def_levels, values


def decompress_page(raw: bytes, expected_size: int) -> bytes:
    return SNAPPY_CODEC.decompress(raw, expected_size).to_pybytes()


def build_scalar(def_levels: list[int], values: list[object]) -> list[object]:
    out = []
    value_idx = 0
    for level in def_levels:
        if level == 1:
            out.append(values[value_idx])
            value_idx += 1
        else:
            out.append(None)
    return out


def build_list(rep_levels: list[int], def_levels: list[int], values: list[int]) -> list[object]:
    rows = []
    current = None
    value_idx = 0

    for rep_level, def_level in zip(rep_levels, def_levels):
        if rep_level == 0:
            if current is not None:
                rows.append(current)
            current = None
        if def_level == 0:
            current = None
            continue
        if current is None:
            current = []
        if def_level == 3:
            current.append(values[value_idx])
            value_idx += 1

    if current is not None:
        rows.append(current)
    return rows


def build_nested_list(rep_levels: list[int], def_levels: list[int], values: list[int]) -> list[object]:
    rows = []
    current_outer = None
    current_inner = None
    value_idx = 0

    for rep_level, def_level in zip(rep_levels, def_levels):
        if rep_level == 0:
            if current_outer is not None:
                if current_inner is not None:
                    current_outer.append(current_inner)
                rows.append(current_outer)
            current_outer = None
            current_inner = None

        if def_level == 0:
            current_outer = None
            current_inner = None
            continue

        if current_outer is None:
            current_outer = []
        if rep_level <= 1:
            if current_inner is not None:
                current_outer.append(current_inner)
            current_inner = []

        if def_level == 5:
            current_inner.append(values[value_idx])
            value_idx += 1

    if current_outer is not None:
        if current_inner is not None:
            current_outer.append(current_inner)
        rows.append(current_outer)
    return rows


def build_column(spec: ColumnSpec, chunk: ColumnChunk) -> list[object]:
    if spec.builder == "scalar":
        return build_scalar(chunk.def_levels, chunk.values)
    if spec.builder == "list":
        return build_list(chunk.rep_levels, chunk.def_levels, chunk.values)
    if spec.builder == "nested_list":
        return build_nested_list(chunk.rep_levels, chunk.def_levels, chunk.values)
    raise ValueError(f"Unknown builder: {spec.builder}")


def parse_column_chunk(data: bytes, pos: int, spec: ColumnSpec, expected_rows: int | None) -> ColumnChunk:
    dictionary = None
    rep_levels: list[int] = []
    def_levels: list[int] = []
    values: list[object] = []
    row_count = 0

    while pos < len(data):
        header, data_pos = parse_page_header(data, pos)
        if header.compressed_size is None or header.uncompressed_size is None:
            raise ValueError("Invalid page header")

        page_end = data_pos + header.compressed_size
        payload = decompress_page(data[data_pos:page_end], header.uncompressed_size)

        if header.page_type == PAGE_TYPE_DICTIONARY_PAGE:
            dictionary = decode_dictionary_page(payload, header, spec.physical_type)
            pos = page_end
            continue

        if header.page_type != PAGE_TYPE_DATA_PAGE:
            raise ValueError(f"Unsupported page type: {header.page_type}")

        page_rep, page_def, page_values = decode_data_page(
            payload,
            header,
            spec.max_rep,
            spec.max_def,
            dictionary,
        )
        rep_levels.extend(page_rep)
        def_levels.extend(page_def)
        values.extend(page_values)
        if spec.max_rep == 0:
            row_count += len(page_def)
        else:
            row_count += sum(level == 0 for level in page_rep)
        pos = page_end

        if expected_rows is None and spec.max_rep == 0:
            expected_rows = len(page_def)
        if expected_rows is not None and row_count >= expected_rows:
            break

    if expected_rows is None:
        raise ValueError(f"Cannot infer row count for repeated column {spec.name}")

    return ColumnChunk(pos, expected_rows, rep_levels, def_levels, values)


def parse_row_group(data: bytes, pos: int) -> tuple[int, pa.Table]:
    columns: dict[str, list[object]] = {}
    row_count: int | None = None

    for spec in COLUMN_SPECS:
        chunk = parse_column_chunk(data, pos, spec, row_count)
        columns[spec.name] = build_column(spec, chunk)
        row_count = chunk.row_count
        pos = chunk.next_pos

    if row_count is None or not all(len(column) == row_count for column in columns.values()):
        raise ValueError("Row group parse mismatch")

    return pos, pa.table(columns, schema=SCHEMA)


def infer_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.recovered.parquet")


def recover_footerless_parquet_to_parquet(path: Path, output: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if raw[:4] != b"PAR1":
        raise ValueError("Not a parquet file")

    pos = 4
    row_groups = 0
    total_rows = 0
    progress = Progress(
        TextColumn("[bold blue]Recovering"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[rows]} rows"),
        TextColumn("{task.fields[row_groups]} row-groups"),
        TimeElapsedColumn(),
        console=CONSOLE,
    )

    with progress, pq.ParquetWriter(output, SCHEMA) as writer:
        task = progress.add_task("recover", total=len(raw), rows=0, row_groups=0)
        while pos + 8 < len(raw):
            try:
                next_pos, table = parse_row_group(raw, pos)
            except Exception:
                break

            if next_pos <= pos or table.num_rows == 0:
                break

            writer.write_table(table)
            pos = next_pos
            row_groups += 1
            total_rows += table.num_rows
            progress.update(
                task,
                completed=min(pos, len(raw)),
                rows=total_rows,
                row_groups=row_groups,
            )

        progress.update(
            task,
            completed=min(pos, len(raw)),
            rows=total_rows,
            row_groups=row_groups,
        )

    if row_groups == 0:
        raise RuntimeError("Failed to recover any row groups, output file may be invalid")

    return total_rows, row_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    output = args.output or infer_output_path(args.path)
    total_rows, row_groups = recover_footerless_parquet_to_parquet(args.path, output)
    CONSOLE.print(f"rows={total_rows}")
    CONSOLE.print(f"row_groups={row_groups}")
    CONSOLE.print(f"saved={output}")


if __name__ == "__main__":
    main()
