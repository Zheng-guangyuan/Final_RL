# Final_RL

This repository contains the implementation and experiments for multi-agent reinforcement learning algorithms, including **CPPO**, **IPPO**, and **MAPPO**, tested on the **VMAS balance scenario**.

## File Structure

- **docs/**: Documentation files for the project.
- **vmas/**: VMAS environment and simulation files.
- **CITATION.cff**: Citation file for referencing this project.
- **README.md**: This README file providing an overview of the project.
- **codecov.yml**: Configuration file for code coverage tools.
- **requirements.txt**: List of Python dependencies required for this project.
- **setup.cfg**: Configuration file for Python packaging.
- **setup.py**: Script for installing the project as a Python package.
- **test_CPPO.py**: Implementation and tests for the CPPO algorithm.
- **test_IPPO.py**: Implementation and tests for the IPPO algorithm.
- **test_MAPPO.py**: Implementation and tests for the MAPPO algorithm.
- **test_PPO.py**: Baseline implementation and tests for PPO.

## Prerequisites

- Python 3.8 or higher
- Ray 2.1.0
- VMAS environment (included in the repository)
- Additional dependencies listed in `requirements.txt`

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Algorithms

Each of the test scripts (`test_CPPO.py`, `test_IPPO.py`, `test_MAPPO.py`, `test_PPO.py`) can be executed to train and evaluate the corresponding algorithm in the **VMAS balance scenario**.

For example, to run the CPPO algorithm:

```bash
python test_CPPO.py
```

### Configurations

- Modify the environment and training parameters directly in the respective test scripts.
- Training logs and evaluation results will be displayed in the console or logged via tools like `wandb` (if configured).

### Environment

The VMAS environment is a vectorized multi-agent simulation framework for reinforcement learning. This repository uses the **balance scenario** with customizable configurations.

