import hashlib
from pathlib import Path

def generate_checksums():
    root_dir = Path(__file__).resolve().parents[3]
    raw_dir = root_dir / "data" / "raw"
    checksums_file = raw_dir / "checksums.sha256"
    
    files_to_hash = []
    for filepath in raw_dir.rglob("*"):
        if filepath.is_file() and filepath.name != "checksums.sha256":
            files_to_hash.append(filepath)
            
    with open(checksums_file, "w") as out_f:
        for filepath in sorted(files_to_hash):
            hasher = hashlib.sha256()
            try:
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096 * 1024), b""):
                        hasher.update(chunk)
                rel_path = filepath.relative_to(raw_dir)
                out_f.write(f"{hasher.hexdigest()}  {rel_path.as_posix()}\n")
                print(f"Hashed: {rel_path.as_posix()}")
            except Exception as e:
                print(f"Error hashing {filepath}: {e}")

if __name__ == "__main__":
    print("Generating checksums...")
    generate_checksums()
    print("Checksums written to data/raw/checksums.sha256")
