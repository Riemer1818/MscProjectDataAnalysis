# MscProjectDataAnalysis

This repository contains tools and scripts designed for analyzing and modeling biological data, with specific components for graph-based models and neural networks. It includes shell scripts, Python and R files, and Jupyter notebooks that facilitate the setup, analysis, and visualization of different data-driven models.

## Repository Structure

- **Shell Scripts** (`makeGraph.sh`, `makeMetaCells.sh`, `runGraphModel.sh`, `runNeuralNetwork.sh`, etc.)
  - Contains bash scripts to automate various stages of data processing and model execution.

- **Python Scripts** (in `src` directory)
  - `dataAnalysisPipeline.ipynb`: Jupyter notebook for the primary data analysis pipeline.
  - `graphModelFunctions.py`, `graphModels.py`: Modules that define functions and implementations for graph-based modeling.
  - `neuralNetworkFunctions.py`, `neuralNetworks.py`: Modules for creating and training neural network models.
  - `makeTFAPlots.py`, `plotLosses.py`: Scripts for plotting and visualizing model performance metrics.

- **R Scripts** (in `src` and `src/archive` directories)
  - `getVariableGenes.r`, `makeH5adFiles.r`, `mapToEnsembl.r`: R scripts that assist in data preparation and transformation for analysis.
  - `makeMetaCells.r`, `makeGraph.py`: Scripts focused on preparing graph data and metacells for further analysis.

- **Notebooks and Tutorials** (in `src` and `src/archive`)
  - `ATACseq.ipynb`, `DataAnalysisPipeline.ipynb`, and others in the `archive` directory offer preliminary and additional analysis workflows.
  - Tutorials include `CellOracle GRN models.ipynb`, which demonstrates how to build gene regulatory network models using CellOracle.

## Getting Started

1. **Dependencies**: Ensure all required Python and R libraries are installed. Refer to `pythonRequirements.txt` and  `requirements.R` if available, or manually install based on script imports.
2. **Data Preparation**: Use the scripts in `src` to preprocess and prepare data. For example, `makeH5adFiles.r` prepares files in `.h5ad` format.
3. **Run Models**: Execute the models using provided shell scripts:
   - `runGraphModel.sh` to initiate graph-based models.
   - `runNeuralNetwork.sh` or other neural network scripts for deep learning models.
4. **Visualization**: Generate plots and performance metrics using scripts like `makeTFAPlots.py` and `plotLosses.py`.

