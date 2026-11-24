"""Restore the lab's validation examples from a local NEU-DET archive."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

from .evidence import CLASSES, RESULTS, ROOT, read_rows


def convert_xml(data: bytes, width: int, height: int) -> str:
    root = ElementTree.fromstring(data)
    labels = []
    for obj in root.findall("object"):
        category = obj.findtext("name", "").strip()
        if category not in CLASSES:
            raise ValueError(f"Unrecognized NEU-DET class: {category}")
        x1, y1, x2, y2 = (float(obj.findtext(f"bndbox/{key}", "")) for key in ("xmin", "ymin", "xmax", "ymax"))
        if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
            raise ValueError("Invalid Pascal VOC coordinates.")
        # The source uses one-based inclusive coordinates; YOLO uses normalized extents.
        x1, y1 = x1 - 1, y1 - 1
        row = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
               (x2 - x1) / width, (y2 - y1) / height)
        labels.append(f"{CLASSES.index(category)} " + " ".join(f"{number:.8f}" for number in row))
    return "\n".join(dict.fromkeys(labels)) + "\n" if labels else ""


def restore_examples(archive: Path, destination: Path = ROOT / "data" / "processed") -> list[str]:
    wanted = {row["image_name"] for row in read_rows(RESULTS / "validation_error_cases.csv")[:12]}
    manifest = read_rows(ROOT / "results" / "phase2" / "yolo_conversion_manifest.csv")
    entries = {Path(row["yolo_image"]).name: row for row in manifest if row["split"] == "val"}
    restored = []
    with ZipFile(archive) as zip_file:
        members = defaultdict(list)
        for member in zip_file.infolist():
            if not member.is_dir():
                members[PurePosixPath(member.filename).name].append(member)
        for name in sorted(wanted):
            entry = entries[name]
            image_name = Path(entry["source_image"]).name
            image = None
            for member in members[image_name]:
                original = zip_file.read(member)
                if hashlib.sha256(original).hexdigest() == entry["image_sha256"]:
                    image = original
                    break
            xml = members[Path(entry["source_xml"]).name]
            if image is None or len(xml) != 1:
                raise ValueError(f"The archive did not contain a verified image and unique annotation for {name}.")
            label = convert_xml(zip_file.read(xml[0]), int(entry["image_width"]), int(entry["image_height"]))
            for path, data in (
                (destination / "images" / "val" / name, image),
                (destination / "labels" / "val" / f"{Path(name).stem}.txt", label.encode("utf-8")),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
            restored.append(name)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Local zip obtained from the NEU dataset publisher.")
    args = parser.parse_args()
    print(f"Available validation examples: {len(restore_examples(args.archive))}")


if __name__ == "__main__":
    main()
