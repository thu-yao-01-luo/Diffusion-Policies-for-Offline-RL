#!/bin/bash

# Specify the range of indices (start and end)
start_index=1
end_index=10

# Specify the source folder (folder a) and destination folder (folder b)
source_folder="xcw-http-648f8c5cbd-q847b:/cephfs/dreamfuser/models/logger/toy/sequence_object_centric_mask"
destination_folder="model/logger/toy/sequence_object_centric_mask"

# Loop through the indices and copy the files
for ((i=start_index; i<=end_index; i++)); do
    filename="agent${i}00000.pth"
    source_path="${source_folder}/${filename}"
    destination_path="${destination_folder}/${filename}"
    kubectl cp "$source_path" "$destination_path"

    filename="agent${i}00000_ema.pth"
    source_path="${source_folder}/${filename}"
    destination_path="${destination_folder}/${filename}"
    kubectl cp "$source_path" "$destination_path"
done
