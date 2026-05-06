# FractalSerl — Fractal Symmetries for Sample-Efficient Robotic Learning

[![Discord](https://img.shields.io/discord/1302866684612444190?label=Join%20Us%20on%20Discord&logo=discord&color=7289da)](https://discord.com/invite/bAxjvvJzNM)
[![Notion](https://img.shields.io/badge/Notion-Workspace-000000?logo=notion&logoColor=white)](https://lipscomb-robotics.notion.site/?source=copy_link)
[![Paper](https://img.shields.io/badge/Paper-Frontiers-blue?logo=zenodo&logoColor=white)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1791812/abstract)
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/lippyrobotics/)
[![YouTube](https://img.shields.io/badge/YouTube-Channel-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@lippyrobotics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Short description
-----------------

FractalSERL implements Branched Euclidean Group Fractal Symmetries — a trajectory-level augmentation framework that accelerates policy learning by iteratively applying affine and Euclidean-group transformations to episodic trajectories. Treating an episodic MDP as a tree of state–action pairs, self-similar branching produces fractal symmetry expansions that populate replay buffers with diverse, consistent experiences. We demonstrate improvements on simulated and real Franka manipulation tasks, achieving robust policies in as little as 14 minutes (avg. ~20 minutes) of wall-clock training.

Contributions in this repo include:
- TODO
- TODO

<figure>
      <img src="docs/images/fetch_rewards_function_grid_size.png" alt="SymmGrid returns as a function of grid size" width="100%">
</figure>

<figure>
      <img src="docs/images/fractal_grid.png" alt="Fractal grid enclosed by the blue blobs" width="100%">
</figure>

Links to docs
--------------

The `docs/` folder contains additional Markdown files with step-by-step guides. Quick links are provided below:

- [Overview of code structure](docs/overview.md)
- [Installation guide](docs/installation.md)
- [Run in simulation](docs/run_sim.md)
- [Run on the real robot](docs/run_realrobot.md)


Quick start (very short)
------------------------

1. Install dependencies: see `docs/installation.md`.
2. Run a demo in sim: see `docs/run_sim.md` for instructions to launch `franka_sim`
3. For real hardware, follow the instructions in `docs/run_realrobot.md` and configure the files related to `serl_robot_infra/`.

Citation
--------

If you use FractalSERL in your research, please cite our paper:

```bibtex
@misc{vanderstelt2026SymmGrid,
      title={Towards Accelerating Deep Reinforcement Learning via Branched Symmetries},
      author={Ryan Vanderstelt, Cleiver Ruiz Martinez, Caeden Rosen, Blake Hull, and Juan Rojas},
      year={2026},
      eprint={____},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```