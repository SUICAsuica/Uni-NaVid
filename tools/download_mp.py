#!/usr/bin/env python3
"""Download Matterport3D release files with Python 3.

This is a Python 3 compatible version of the Matterport3D public data release
script. Only download the file types you need; the complete release is large.
"""

import argparse
import os
import tempfile
import urllib.request

BASE_URL = "http://kaldir.vc.cit.tum.de/matterport/"
RELEASE = "v1/scans"
RELEASE_TASKS = "v1/tasks"
RELEASE_SIZE = "1.3TB"
TOS_URL = BASE_URL + "MP_TOS.pdf"

FILETYPES = [
    "cameras",
    "matterport_camera_intrinsics",
    "matterport_camera_poses",
    "matterport_color_images",
    "matterport_depth_images",
    "matterport_hdr_images",
    "matterport_mesh",
    "matterport_skybox_images",
    "undistorted_camera_parameters",
    "undistorted_color_images",
    "undistorted_depth_images",
    "undistorted_normal_images",
    "house_segmentations",
    "region_segmentations",
    "image_overlap_data",
    "poisson_meshes",
    "sens",
]

TASK_FILES = {
    "keypoint_matching_data": ["keypoint_matching/data.zip"],
    "keypoint_matching_models": ["keypoint_matching/models.zip"],
    "surface_normal_data": ["surface_normal/data_list.zip"],
    "surface_normal_models": ["surface_normal/models.zip"],
    "region_classification_data": ["region_classification/data.zip"],
    "region_classification_models": ["region_classification/models.zip"],
    "semantic_voxel_label_data": ["semantic_voxel_label/data.zip"],
    "semantic_voxel_label_models": ["semantic_voxel_label/models.zip"],
    "minos": ["mp3d_minos.zip"],
    "gibson": ["mp3d_for_gibson.tar.gz"],
    "habitat": ["mp3d_habitat.zip"],
    "pixelsynth": ["mp3d_pixelsynth.zip"],
    "igibson": ["mp3d_for_igibson.zip"],
    "mp360": [
        "mp3d_360/data_00.zip",
        "mp3d_360/data_01.zip",
        "mp3d_360/data_02.zip",
        "mp3d_360/data_03.zip",
        "mp3d_360/data_04.zip",
        "mp3d_360/data_05.zip",
        "mp3d_360/data_06.zip",
    ],
}


def get_release_scans(release_file):
    with urllib.request.urlopen(release_file) as response:
        return [line.decode("utf-8").strip() for line in response if line.strip()]


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(out_file):
        print(f"WARNING: skipping existing file {out_file}")
        return

    print(f"\t{url} > {out_file}")
    fd, tmp_file = tempfile.mkstemp(dir=out_dir)
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp_file)
        os.rename(tmp_file, out_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def download_scan(scan_id, out_dir, file_types):
    print(f"Downloading MP scan {scan_id} ...")
    os.makedirs(out_dir, exist_ok=True)
    for file_type in file_types:
        url = f"{BASE_URL}{RELEASE}/{scan_id}/{file_type}.zip"
        download_file(url, os.path.join(out_dir, f"{file_type}.zip"))
    print(f"Downloaded scan {scan_id}")


def download_release(scan_ids, out_dir, file_types):
    print(f"Downloading MP release to {out_dir} ...")
    for scan_id in scan_ids:
        download_scan(scan_id, os.path.join(out_dir, scan_id), file_types)
    print("Downloaded MP release.")


def download_task_data(task_data, out_dir):
    print(f"Downloading MP task data for {task_data} ...")
    for task_data_id in task_data:
        for file_part in TASK_FILES[task_data_id]:
            url = f"{BASE_URL}{RELEASE_TASKS}/{file_part}"
            download_file(url, os.path.join(out_dir, file_part))
    print(f"Done downloading task data for {task_data}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Matterport3D public data release.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-o", "--out_dir", required=True, help="download directory")
    parser.add_argument("--id", default="ALL", help="scan id, or ALL")
    parser.add_argument("--type", nargs="+", default=None, help="file types to download")
    parser.add_argument("--task_data", nargs="+", default=[], help="task data ids to download")
    parser.add_argument("--yes", action="store_true", help="skip interactive TOS prompts")
    args = parser.parse_args()

    file_types = args.type or FILETYPES
    invalid_types = sorted(set(file_types) - set(FILETYPES))
    invalid_tasks = sorted(set(args.task_data) - set(TASK_FILES))
    if invalid_types:
        raise SystemExit(f"Invalid file type(s): {', '.join(invalid_types)}")
    if invalid_tasks:
        raise SystemExit(f"Invalid task data id(s): {', '.join(invalid_tasks)}")

    print("Continuing confirms that you have agreed to the MP terms of use:")
    print(TOS_URL)
    if not args.yes:
        input("Press Enter to continue, or CTRL-C to exit.")

    if args.task_data:
        download_task_data(args.task_data, os.path.join(args.out_dir, RELEASE_TASKS))

    release_scans = get_release_scans(BASE_URL + RELEASE + ".txt")
    scan_id = args.id
    if scan_id and scan_id.lower() != "all":
        if scan_id not in release_scans:
            raise SystemExit(f"Invalid scan id: {scan_id}")
        download_scan(scan_id, os.path.join(args.out_dir, RELEASE, scan_id), file_types)
        return

    if len(file_types) == len(FILETYPES):
        print(f"WARNING: the entire MP release requires about {RELEASE_SIZE}.")
    else:
        print(f"WARNING: downloading all MP scans for: {', '.join(file_types)}")
    if not args.yes:
        input("Press Enter to continue, or CTRL-C to exit.")
    download_release(release_scans, os.path.join(args.out_dir, RELEASE), file_types)


if __name__ == "__main__":
    main()
