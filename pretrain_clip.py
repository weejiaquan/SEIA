"""
Pre-train CLIP Embedding Caches for All Scripts

Scans all script folders for template directories and generates
embedding caches in each script's cache/ folder.

After running, you can:
1. Commit the cache/ folders to git
2. Add templates/ to .gitignore
3. Share scripts with just the cache (no images needed)

Usage:
    python pretrain_clip.py                         # all scripts
    python pretrain_clip.py scripts/character_scan  # specific script
    python pretrain_clip.py --force                 # rebuild all caches
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.clip_scanner import build_embedding_cache


def is_cache_current(cache_file: str, template_folder: str) -> bool:
    """
    Check if cache is up to date by comparing modification times.

    Returns True if cache exists and is newer than all template files.
    """
    if not os.path.exists(cache_file):
        return False

    cache_mtime = os.path.getmtime(cache_file)

    # Check all PNG files in template folder
    for f in os.listdir(template_folder):
        if f.lower().endswith('.png'):
            png_path = os.path.join(template_folder, f)
            if os.path.getmtime(png_path) > cache_mtime:
                return False  # Template is newer than cache

    return True


def find_template_sources(script_dir: str) -> list[tuple[str, str, str]]:
    """
    Find all template sources in a script directory.

    Returns list of (path, cache_name, type) tuples where:
    - path: folder path or templates_dir for root PNGs
    - cache_name: name for the cache file
    - type: "folder" or "root"
    """
    templates_dir = os.path.join(script_dir, "templates")
    if not os.path.exists(templates_dir):
        return []

    results = []

    # Check for subfolders with PNGs
    for item in os.listdir(templates_dir):
        item_path = os.path.join(templates_dir, item)
        if os.path.isdir(item_path):
            pngs = [f for f in os.listdir(item_path) if f.lower().endswith('.png')]
            if pngs:
                results.append((item_path, item, "folder"))

    # Check for root-level PNGs -> single "templates" cache
    root_pngs = [f for f in os.listdir(templates_dir) if f.lower().endswith('.png')]
    if root_pngs:
        results.append((templates_dir, "templates", "root"))

    return results


def pretrain_script(script_dir: str, force: bool = False) -> int:
    """
    Pre-train all templates in a script directory.

    Structure:
    - templates/*.png → cache/templates.npz (root PNGs grouped)
    - templates/subfolder/*.png → cache/subfolder.npz

    Returns number of caches generated.
    """
    script_name = os.path.basename(script_dir)
    template_sources = find_template_sources(script_dir)

    if not template_sources:
        print(f"  No templates found in {script_dir}")
        return 0

    # Create cache directory
    cache_dir = os.path.join(script_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    count = 0
    skipped = 0

    for template_path, cache_name, source_type in template_sources:
        # Count PNGs (only direct children, not recursive for root)
        if source_type == "root":
            pngs = [f for f in os.listdir(template_path) if f.lower().endswith('.png') and os.path.isfile(os.path.join(template_path, f))]
        else:
            pngs = [f for f in os.listdir(template_path) if f.lower().endswith('.png')]

        png_count = len(pngs)
        cache_file = os.path.join(cache_dir, f"{cache_name}.npz")

        # Check if cache is already up to date
        if not force and is_cache_current(cache_file, template_path):
            cache_size = os.path.getsize(cache_file)
            print(f"  [{cache_name}] Up to date ({png_count} templates, {cache_size / 1024:.1f} KB) - skipped")
            skipped += 1
            continue

        print(f"\n  [{cache_name}] {png_count} templates - building...")

        try:
            # For root templates, don't recurse into subfolders
            recursive = (source_type != "root")
            cache = build_embedding_cache(
                template_folder=template_path,
                cache_file=cache_file,
                batch_size=16,
                recursive=recursive,
            )

            cache_size = os.path.getsize(cache_file)
            print(f"  [{cache_name}] Cache saved: {cache_file} ({cache_size / 1024:.1f} KB)")
            count += 1

        except Exception as e:
            print(f"  [{cache_name}] ERROR: {e}")

    if skipped > 0:
        print(f"\n  Skipped {skipped} up-to-date cache(s)")

    return count


def main() -> None:
    print("=" * 60)
    print("CLIP Embedding Cache Pre-trainer")
    print("=" * 60)

    # Parse arguments
    args = sys.argv[1:]
    force = "--force" in args or "-f" in args
    args = [a for a in args if a not in ("--force", "-f")]

    if force:
        print("\n[Force rebuild enabled]")

    # Check for specific script argument
    if args:
        script_dirs = [args[0]]
    else:
        # Find all script directories
        scripts_root = os.path.join(os.path.dirname(__file__), "scripts")
        if not os.path.exists(scripts_root):
            print("ERROR: scripts/ directory not found")
            sys.exit(1)

        script_dirs = [
            os.path.join(scripts_root, d)
            for d in os.listdir(scripts_root)
            if os.path.isdir(os.path.join(scripts_root, d))
            and not d.startswith('.')
        ]

    if not script_dirs:
        print("No script directories found")
        sys.exit(1)

    print(f"\nFound {len(script_dirs)} script folder(s)")

    total_caches = 0
    for script_dir in sorted(script_dirs):
        script_name = os.path.basename(script_dir)
        print(f"\n{'='*60}")
        print(f"Script: {script_name}")
        print("-" * 60)

        count = pretrain_script(script_dir, force=force)
        total_caches += count

    print(f"\n{'='*60}")
    print("COMPLETE")
    print("=" * 60)
    print(f"Generated {total_caches} cache(s)")
    print()
    print("Next steps:")
    print("  1. Add to .gitignore:")
    print("     scripts/*/templates/**/*.png")
    print()
    print("  2. Commit cache folders:")
    print("     git add scripts/*/cache/*.npz")
    print()
    print("  3. Remove images from git (optional):")
    print("     git rm --cached 'scripts/*/templates/**/*.png'")
    print()


if __name__ == "__main__":
    main()
