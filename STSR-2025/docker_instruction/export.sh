#!/usr/bin/env bash

# This script exports the built Docker image to a .tar.gz file for submission.
# --> ACTION REQUIRED: Replace 'teamname' with your actual team name.
TEAM_NAME="teamname"

IMAGE_NAME="${TEAM_NAME}:latest"
OUTPUT_FILE="${TEAM_NAME}.tar.gz"

echo "Exporting Docker image ${IMAGE_NAME} to ${OUTPUT_FILE}"

docker save "${IMAGE_NAME}" | gzip -c > "${OUTPUT_FILE}"

echo "Export complete. Please submit the file: ${OUTPUT_FILE}"