#!/usr/bin/env bash

# This script builds the Docker image.
# The image will be tagged with 'teamname:latest'.
# --> ACTION REQUIRED: Replace 'teamname' with your actual team name.
TEAM_NAME="teamname"

echo "Building Docker image: ${TEAM_NAME}:latest"

docker build -t "${TEAM_NAME}:latest" .