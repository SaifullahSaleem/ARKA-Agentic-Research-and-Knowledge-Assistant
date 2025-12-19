"""
Monitoring Setup for Prometheus and Grafana
Creates dashboards and configures metrics collection.
"""

import json
import os
from typing import Dict

# Grafana dashboard JSON structure
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Agentic AI Interface Metrics",
        "tags": ["agentic-ai", "research-papers"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "Query Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(agent_queries_total[5m])",
                        "legendFormat": "Queries/sec",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            {
                "id": 2,
                "title": "Query Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, agent_query_latency_seconds_bucket)",
                        "legendFormat": "95th percentile",
                    },
                    {
                        "expr": "histogram_quantile(0.50, agent_query_latency_seconds_bucket)",
                        "legendFormat": "50th percentile",
                    },
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            {
                "id": 3,
                "title": "Tokens Consumed",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(agent_tokens_total[5m])",
                        "legendFormat": "Tokens/sec",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            },
            {
                "id": 4,
                "title": "Chunks Retrieved",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, agent_chunks_retrieved_bucket)",
                        "legendFormat": "95th percentile",
                    },
                    {
                        "expr": "avg(agent_chunks_retrieved)",
                        "legendFormat": "Average",
                    },
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            },
            {
                "id": 5,
                "title": "Active Queries",
                "type": "singlestat",
                "targets": [
                    {
                        "expr": "agent_active_queries",
                    }
                ],
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
            },
            {
                "id": 6,
                "title": "Total Queries",
                "type": "singlestat",
                "targets": [
                    {
                        "expr": "agent_queries_total",
                    }
                ],
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
            },
        ],
        "refresh": "10s",
        "schemaVersion": 16,
        "version": 1,
    }
}


def create_prometheus_config():
    """Create Prometheus configuration file."""
    config = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s",
        },
        "scrape_configs": [
            {
                "job_name": "agentic-ai",
                "static_configs": [
                    {
                        "targets": ["localhost:9090"],
                    }
                ],
            }
        ],
    }
    
    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/prometheus.yml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created monitoring/prometheus.yml")


def create_grafana_dashboard():
    """Create Grafana dashboard JSON file."""
    os.makedirs("monitoring/grafana/dashboards", exist_ok=True)
    
    with open("monitoring/grafana/dashboards/agentic_ai.json", "w") as f:
        json.dump(GRAFANA_DASHBOARD, f, indent=2)
    
    print("Created monitoring/grafana/dashboards/agentic_ai.json")


def create_docker_compose():
    """Create docker-compose.yml for Prometheus and Grafana."""
    docker_compose = """version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - monitoring
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("Created docker-compose.yml")


def main():
    """Setup monitoring infrastructure."""
    print("=" * 60)
    print("Setting up Monitoring Infrastructure")
    print("=" * 60)
    
    create_prometheus_config()
    create_grafana_dashboard()
    create_docker_compose()
    
    print("\n" + "=" * 60)
    print("Monitoring setup complete!")
    print("=" * 60)
    print("\nTo start monitoring:")
    print("1. docker-compose up -d")
    print("2. Access Prometheus at http://localhost:9090")
    print("3. Access Grafana at http://localhost:3000 (admin/admin)")
    print("4. Import dashboard from monitoring/grafana/dashboards/agentic_ai.json")


if __name__ == "__main__":
    main()

