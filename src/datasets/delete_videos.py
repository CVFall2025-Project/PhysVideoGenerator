#!/usr/bin/env python3
"""
Video File Selector
This script groups video files by their main name and allows you to keep
a specified number of videos from each group, deleting the rest.
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def parse_filename(filename):
    """
    Extract the main name, main_id, and sub_id from a filename.
    Assumes format: main_name_main_id_sub_id.mp4
    where main_name can contain underscores (e.g., pixabay_seascapes_videos_123_456.mp4)
    Returns: (main_name, main_id, sub_id, full_filename)
    """
    # Remove extension
    name_without_ext = Path(filename).stem
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # We need at least 3 parts: main_name, main_id, sub_id
    if len(parts) >= 2:
        # Last part should be sub_id
        # Second to last should be main_id
        # Everything before that is main_name
        try:
            sub_id = int(parts[-1])
            main_id = int(parts[-2])
            # Join all parts before the last two as main_name
            main_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
            return main_name, main_id, sub_id, filename
        except ValueError:
            # If conversion fails, treat entire name as main_name
            return name_without_ext, 0, 0, filename
    else:
        # If not enough parts, treat entire name as main_name
        return name_without_ext, 0, 0, filename


def group_videos(folder_path):
    """
    Group video files by their main name.
    Only considers .mp4 files.
    
    Args:
        folder_path: Path to the folder containing videos
    
    Returns:
        Dictionary with main_name as key and list of (main_id, sub_id, filename) as value
    """
    groups = defaultdict(list)
    
    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
        
        # Check if file is .mp4
        if Path(filename).suffix.lower() == '.mp4':
            main_name, main_id, sub_id, full_filename = parse_filename(filename)
            groups[main_name].append((main_id, sub_id, full_filename))
    
    # Sort each group by main_id, then sub_id
    for main_name in groups:
        groups[main_name].sort(key=lambda x: (x[0], x[1]))
    
    return groups


def select_videos(groups, num_to_keep, selection_method='first'):
    """
    Select which videos to keep from each group.
    
    Args:
        groups: Dictionary of grouped videos
        num_to_keep: Number of videos to keep from each group
        selection_method: 'first', 'last', 'random', or 'evenly_spaced'
    
    Returns:
        Tuple of (files_to_keep, files_to_delete)
    """
    import random
    
    files_to_keep = []
    files_to_delete = []
    
    for main_name, videos in groups.items():
        if len(videos) <= num_to_keep:
            # Keep all if we have fewer than requested
            files_to_keep.extend([v[2] for v in videos])  # v[2] is filename
        else:
            if selection_method == 'first':
                keep = videos[:num_to_keep]
                delete = videos[num_to_keep:]
            elif selection_method == 'last':
                keep = videos[-num_to_keep:]
                delete = videos[:-num_to_keep]
            elif selection_method == 'random':
                keep = random.sample(videos, num_to_keep)
                delete = [v for v in videos if v not in keep]
            elif selection_method == 'evenly_spaced':
                # Select evenly spaced indices
                indices = [int(i * len(videos) / num_to_keep) for i in range(num_to_keep)]
                keep = [videos[i] for i in indices]
                delete = [v for i, v in enumerate(videos) if i not in indices]
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            
            files_to_keep.extend([v[2] for v in keep])  # v[2] is filename
            files_to_delete.extend([v[2] for v in delete])  # v[2] is filename
    
    return files_to_keep, files_to_delete


def main():
    parser = argparse.ArgumentParser(
        description='Select and keep a specific number of videos from each group, delete the rest.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep first 5 videos from each main_name group (dry run)
  python video_selector.py /path/to/videos --keep 5
  
  # Keep first 5 videos and actually delete the rest
  python video_selector.py /path/to/videos --keep 5 --delete
  
  # Keep last 3 videos from each group
  python video_selector.py /path/to/videos --keep 3 --method last
  
  # Keep 10 evenly spaced videos from each group
  python video_selector.py /path/to/videos --keep 10 --method evenly_spaced

Filename format expected: main_name_main_id_sub_id.mp4
  Example: pixabay_seascapes_videos_123_456.mp4
  - main_name: pixabay_seascapes_videos
  - main_id: 123
  - sub_id: 456
        """
    )
    
    parser.add_argument('folder', help='Path to folder containing .mp4 video files')
    parser.add_argument('--keep', '-k', type=int, required=True,
                        help='Number of videos to keep from each group')
    parser.add_argument('--method', '-m', choices=['first', 'last', 'random', 'evenly_spaced'],
                        default='first', help='Method for selecting videos (default: first)')
    parser.add_argument('--delete', '-d', action='store_true',
                        help='Actually delete files (without this flag, only shows what would be deleted)')
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory")
        return
    
    # Group videos
    print(f"Scanning folder: {args.folder}")
    groups = group_videos(args.folder)
    
    if not groups:
        print("No .mp4 video files found!")
        return
    
    print(f"\nFound {len(groups)} video groups:")
    for main_name, videos in sorted(groups.items()):
        print(f"  {main_name}: {len(videos)} videos")
    
    # Select videos
    files_to_keep, files_to_delete = select_videos(groups, args.keep, args.method)
    
    print(f"\n{'='*60}")
    print(f"Selection method: {args.method}")
    print(f"Videos to keep per group: {args.keep}")
    print(f"Total videos to keep: {len(files_to_keep)}")
    print(f"Total videos to delete: {len(files_to_delete)}")
    print(f"{'='*60}\n")
    
    # Show what will be deleted
    if files_to_delete:
        print("Files to be deleted:")
        for filename in sorted(files_to_delete):
            print(f"  - {filename}")
        print()
    
    # Delete files if requested
    if args.delete:
        confirm = input(f"Are you sure you want to delete {len(files_to_delete)} files? (yes/no): ")
        if confirm.lower() == 'yes':
            deleted_count = 0
            for filename in files_to_delete:
                file_path = os.path.join(args.folder, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
            print(f"\nSuccessfully deleted {deleted_count} files")
        else:
            print("Deletion cancelled")
    else:
        print("DRY RUN - No files were deleted. Use --delete flag to actually delete files.")


if __name__ == '__main__':
    main()