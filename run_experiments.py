"""
Script to run evaluation experiments
"""

import json
import pandas as pd
from experimentation_suite import ExperimentationSuite
from evaluation_metrics import EvaluationMetrics

def main():
    """Run evaluation experiments."""
    print("=" * 60)
    print("Running Evaluation Experiments")
    print("=" * 60)
    
    suite = ExperimentationSuite()
    
    # Run with a subset for testing (remove num_queries for full run)
    results_df = suite.run_experiments(num_queries=10)
    
    # Analyze
    analysis = suite.analyze_results(results_df)
    
    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()

