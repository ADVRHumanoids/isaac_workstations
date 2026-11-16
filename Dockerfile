# syntax=docker/dockerfile:1
FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential cmake libglib2.0-0 libgl1 libegl1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN git clone --depth 1 --branch v3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab
WORKDIR /opt/IsaacLab
RUN uv venv --python 3.12 --seed env_isaaclab
ENV VIRTUAL_ENV=/opt/IsaacLab/env_isaaclab
ENV LD_LIBRARY_PATH=/opt/IsaacLab/env_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN ./isaaclab.sh -i && uv cache clean

# X11 renderer dependency.
RUN apt-get update && apt-get install -y --no-install-recommends libxt6 libxrender1 libxrandr2 libxi6 libxcursor1 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*
# Isaac Sim runtime (PhysX and Omniverse Kit).
RUN uv pip install 'isaacsim[all,extscache]==6.0.0.1' \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow \
    && uv cache clean

RUN uv pip install 'wandb==0.22.3' 'websockets==13.1' && uv cache clean

RUN <<'SH'
git -C /opt/IsaacLab apply --unidiff-zero <<'PATCH'
--- a/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
+++ b/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
@@ -1842 +1842 @@
-        scene_data_provider.get_transforms(cls._scene_data, mapping=cls._scene_data_mapping)
+        scene_data_provider.get_transforms(cls._scene_data, mapping=cls._scene_data_mapping, allow_passthrough=False)
PATCH
SH

WORKDIR /workspace
CMD ["bash"]
