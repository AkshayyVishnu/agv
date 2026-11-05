# Offline mapping on the laptop: slam_toolbox over a recorded bag (scripts/make_map.sh).
#   docker build -t turing/mapping:humble -f docker/mapping.Dockerfile docker
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-slam-toolbox ros-humble-nav2-map-server python3-numpy \
    && rm -rf /var/lib/apt/lists/*
