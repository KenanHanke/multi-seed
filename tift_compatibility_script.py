#!/usr/bin/env python3

from pathlib import Path
import gzip

# Get this script's location as it is written to the results dir
script_dir = Path(__file__).resolve().parent

# Delete all .mat.gz files
for file in script_dir.rglob("*.mat.gz"):
    file.unlink()

# Rename all .img.gz files to .img.z
for file in script_dir.rglob("*.img.gz"):
    new_name = file.with_suffix('.z')
    file.rename(new_name)

# Extract all .hdr.gz files destructively
for file in script_dir.rglob("*.hdr.gz"):
    new_name = file.with_suffix('') # Remove the .gz suffix
    with gzip.open(file.as_posix(), 'rb') as f_in:
        with open(new_name.as_posix(), 'wb') as f_out:
            f_out.write(f_in.read())
    file.unlink()  # Remove the original .hdr.gz file

# Write the .prj files
mask_file = script_dir / "mask.hdr"

for network_folder in script_dir.glob("network_*"):
    if not network_folder.is_dir():
        continue
    
    visualization_file = network_folder / "average_visualization.hdr"
    prj_file = network_folder / (network_folder.name + ".prj")
    
    cohort_folders = [x for x in network_folder.iterdir() if x.is_dir()]
    
    prj_content = []

    for i, cohort_folder in enumerate(cohort_folders, start=1):
        dti_name = cohort_folder / "result_for_dataset_000001.hdr"
        prj_line = "DTINAME%d: %s" % (i, dti_name.as_posix())
        prj_content.append(prj_line)
    
    # These lines need to be inserted after the first DTINAME line
    prj_content[1:1] = [
        "MASKNAME: %s" % mask_file.as_posix(),
        "BGNAME:   %s" % visualization_file.as_posix()
    ]
    
    prj_file.write_text("\n".join(prj_content))