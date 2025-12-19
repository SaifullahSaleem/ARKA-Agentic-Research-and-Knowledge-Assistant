"""
Experimentation Suite
Systematically vary hyperparameters and collect evaluation results.
"""

import os
import json
import yaml
import pandas as pd
from typing import Dict, List
from itertools import product
from datetime import datetime
import logging

from agentic_interface import get_interface
from evaluation_metrics import EvaluationMetrics
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentationSuite:
    """Run systematic experiments with different hyperparameter settings."""
    
    def __init__(self):
        self.evaluator = EvaluationMetrics()
        self.test_queries = self._load_test_queries()
        self.results = []
    
    def _load_test_queries(self) -> List[Dict]:
        """Load test queries for evaluation."""
        # Default test queries for Machine Learning in Healthcare
        default_queries = [
            {
                "query": "What are the main applications of deep learning in medical imaging?",
                "domain": "medical imaging",
            },
            {
                "query": "How do neural networks help in clinical decision support systems?",
                "domain": "clinical decision support",
            },
            {
                "query": "What are the challenges of using AI in electronic health records?",
                "domain": "EHR",
            },
            {
                "query": "Explain federated learning approaches in healthcare.",
                "domain": "federated learning",
            },
            {
                "query": "What is the role of transformers in healthcare NLP?",
                "domain": "NLP",
            },
            {
                "query": "How is reinforcement learning used for treatment optimization?",
                "domain": "reinforcement learning",
            },
            {
                "query": "What are the ethical considerations in AI healthcare?",
                "domain": "ethics",
            },
            {
                "query": "Describe computer vision applications in pathology.",
                "domain": "pathology",
            },
            {
                "query": "How do time series models help in patient monitoring?",
                "domain": "time series",
            },
            {
                "query": "What is precision medicine and how does ML support it?",
                "domain": "precision medicine",
            },
        ]
        
        # Try to load from file if exists
        if os.path.exists("test_queries.json"):
            try:
                with open("test_queries.json", "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list) and len(loaded) > 0:
                        return loaded
            except Exception as e:
                logger.warning(f"Could not load test_queries.json: {e}")
        
        return default_queries
    
    def define_parameter_grid(self) -> List[Dict]:
        """
        Define hyperparameter grid for experiments.
        
        Returns:
            List of parameter configurations
        """
        # Load base config
        with open("config.yaml", "r") as f:
            base_config = yaml.safe_load(f)
        
        # Define parameter ranges
        parameter_grid = {
            "k": [3, 5, 7, 10],  # Number of chunks to retrieve
            "chunk_size": [500, 1000, 1500],  # Chunk size
            "temperature": [0.3, 0.7, 1.0],  # LLM temperature
            "model": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"],  # LLM model
        }
        
        # Generate all combinations
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        
        configurations = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            configurations.append(config)
        
        return configurations
    
    def run_single_experiment(
        self,
        config: Dict,
        query: Dict,
        interface
    ) -> Dict:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Hyperparameter configuration
            query: Test query dictionary
            interface: Agentic interface instance
            
        Returns:
            Evaluation results
        """
        # Temporarily update config (in real implementation, would create new interface)
        # For now, we'll use the default interface and log the config
        
        try:
            # Query the interface
            result = interface.query(query["query"])
            
            # Evaluate
            evaluation = self.evaluator.evaluate_query(
                query=query["query"],
                answer=result["answer"],
                retrieved_chunks=result["context_chunks"],
                latency_seconds=result["metadata"]["latency_seconds"],
                tokens_used=result["metadata"]["estimated_tokens"],
            )
            
            # Add configuration info
            evaluation["config"] = config
            evaluation["query_info"] = query
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", extra={"config": config, "query": query})
            return {
                "query": query["query"],
                "config": config,
                "error": str(e),
            }
    
    def run_experiments(
        self,
        num_queries: int = None,
        configs: List[Dict] = None
    ) -> pd.DataFrame:
        """
        Run batch of experiments.
        
        Args:
            num_queries: Number of test queries to use (None = all)
            configs: List of configurations to test (None = use parameter grid)
            
        Returns:
            DataFrame with all results
        """
        if configs is None:
            configs = self.define_parameter_grid()
        
        queries = self.test_queries[:num_queries] if num_queries else self.test_queries
        
        logger.info(f"Starting experiments: {len(configs)} configs × {len(queries)} queries")
        
        interface = get_interface()
        
        all_results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Running config {i+1}/{len(configs)}: {config}")
            
            for j, query in enumerate(queries):
                logger.info(f"  Query {j+1}/{len(queries)}: {query['query'][:50]}...")
                
                result = self.run_single_experiment(config, query, interface)
                all_results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"experiment_results_{timestamp}.json"
        df.to_json(results_file, indent=2, orient="records")
        logger.info(f"Results saved to {results_file}")
        
        # Also save as CSV for easier analysis
        csv_file = f"experiment_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze experiment results and generate insights.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            Analysis summary
        """
        if results_df.empty:
            return {}
        
        # Aggregate by configuration
        analysis = {}
        
        # Group by config
        if "config" in results_df.columns:
            for config_str, group in results_df.groupby("config"):
                config_results = group.to_dict("records")
                aggregated = self.evaluator.evaluate_batch(config_results)
                analysis[str(config_str)] = aggregated
        
        # Overall statistics
        if "context_precision" in results_df.columns:
            analysis["overall"] = {
                "mean_cp": results_df["context_precision"].mean(),
                "mean_faithfulness": results_df["faithfulness"].apply(
                    lambda x: x["faithfulness_score"] if isinstance(x, dict) else 0
                ).mean(),
            }
        
        return analysis


def main():
    """Main function to run experiments."""
    print("=" * 60)
    print("Experimentation Suite")
    print("=" * 60)
    
    suite = ExperimentationSuite()
    
    # Run experiments with a subset for faster testing
    # For full experiments, remove num_queries parameter
    results_df = suite.run_experiments(num_queries=5)  # Use 5 queries for testing
    
    # Analyze results
    analysis = suite.analyze_results(results_df)
    
    # Save analysis
    with open("experiment_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Experiments completed!")
    print("=" * 60)
    print(f"Total experiments: {len(results_df)}")
    print(f"Results saved to experiment_results_*.json and *.csv")


if __name__ == "__main__":
    main()

