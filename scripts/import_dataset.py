"""Import the supplied student archive without overwriting different raw data."""
import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def import_dataset(archive, destination):
    count = 0
    with ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if len(parts) != 4 or parts[1] not in {'train', 'student_test'}:
                raise ValueError(f'Unexpected archive member: {entry.filename}')
            _, partition, trip, filename = parts
            if len(trip) != 8 or not trip.isascii() or not trip.isdigit():
                raise ValueError(f'Invalid trip ID: {trip}')
            if filename not in {f'{trip}.{ext}' for ext in ('track', 'nodes', 'arcs', 'route')}:
                raise ValueError(f'Unexpected trip file: {filename}')
            path = destination / partition / trip / filename
            contents = source.read(entry)
            if path.exists():
                if path.read_bytes() != contents:
                    raise ValueError(f'Existing file differs from archive: {path}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            count += 1
    return count


def import_evaluator(archive, destination, student_root):
    """Validate the held-out partition and store references outside the inference folders."""
    extensions = {'track', 'nodes', 'arcs', 'route'}
    development_ids = {path.name for part in ('train', 'student_test')
                       for path in (student_root / part).iterdir() if path.is_dir()}
    with ZipFile(archive) as source:
        trips = defaultdict(dict)
        for entry in source.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if len(parts) != 3 or parts[0] != 'evaluator_test':
                raise ValueError(f'Unexpected evaluator archive member: {entry.filename}')
            _, trip, filename = parts
            if len(trip) != 8 or not trip.isascii() or not trip.isdigit() or trip in development_ids:
                raise ValueError(f'Invalid or overlapping evaluator trip ID: {trip}')
            ext = PurePosixPath(filename).suffix[1:]
            if ext not in extensions or filename != f'{trip}.{ext}':
                raise ValueError(f'Unexpected evaluator trip file: {filename}')
            if ext in trips[trip] or entry.file_size == 0:
                raise ValueError(f'Empty or duplicate evaluator trip file: {entry.filename}')
            trips[trip][ext] = entry
        if len(trips) != 10 or any(set(files) != extensions for files in trips.values()):
            raise ValueError('Expected ten distinct evaluator trips with all four files')
        for trip, files in sorted(trips.items()):
            for ext, entry in files.items():
                folder = destination / 'references' if ext == 'route' else destination / 'trips' / trip
                path = folder / f'{trip}.{ext}'
                contents = source.read(entry)
                if path.exists():
                    if path.read_bytes() != contents:
                        raise ValueError(f'Existing file differs from archive: {path}')
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(contents)
    return len(trips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--evaluator', action='store_true', help='Import the reserved evaluator partition')
    parser.add_argument('--student-root', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    if args.evaluator:
        output = args.out or Path('data/evaluator')
        count = import_evaluator(args.archive, output, args.student_root)
        print(f'{count} reserved trips verified or imported into {output}, with private references stored separately')
    else:
        output = args.out or args.student_root
        print(f'{import_dataset(args.archive, output)} files verified or imported into {output}')
