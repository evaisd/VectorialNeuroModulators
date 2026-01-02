from pathlib import Path
from neuro_mod.execution.helpers import Logger
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer


def main():
    logger = Logger(name="LoadSNNAnalysis")
    analysis_dir = Path("/Users/eviatar/PycharmProjects/VectorialNeuroModulators/simulations/repeated_long/analysis")
    logger.info(f"Loading analysis from {analysis_dir}.")
    analyzer = SNNAnalyzer(analysis_dir)
    logger.info(f"Loaded {analyzer.get_num_states()} attractors.")
    logger.info(f"Transition matrix shape: {analyzer.get_transition_matrix().shape}")
    return analyzer

if __name__ == "__main__":
    analyzer = main()
