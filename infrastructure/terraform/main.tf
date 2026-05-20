# Yosai MLOps Kubernetes Deployment

terraform {
  required_version = ">= 1.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "yosai-mlops"
}

# Namespace resource
resource "kubernetes_namespace" "this" {
  metadata {
    name = var.namespace
  }
}

# ConfigMap for application configuration
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "yosai-config"
    namespace = var.namespace
  }

  data = {
    MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
    YOSAI_MLFLOW_EXPERIMENT = "yosai-edge-mlops"
    ENVIRONMENT = "production"
  }
}

# Deployment for inference API
resource "kubernetes_deployment" "api" {
  metadata {
    name      = "yosai-api"
    namespace = var.namespace
    labels = {
      app = "yosai-api"
    }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "yosai-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "yosai-api"
        }
      }

      spec {
        container {
          image = "yosai/mlops:${var.image_tag}"
          name  = "api"

          port {
            container_port = 8000
            name           = "http"
          }

          env {
            name = "MLFLOW_TRACKING_URI"
            value_from {
              config_map_key_ref {
                name = "yosai-config"
                key  = "MLFLOW_TRACKING_URI"
              }
            }
          }

          resources {
            requests = {
              cpu    = "500m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1000m"
              memory = "1Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health/live"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/health/ready"
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 5
          }
        }
      }
    }
  }
}

# Service for API
resource "kubernetes_service" "api" {
  metadata {
    name      = "yosai-api"
    namespace = var.namespace
  }

  spec {
    selector = {
      app = "yosai-api"
    }

    port {
      name       = "http"
      port       = 80
      target_port = 8000
    }

    type = "ClusterIP"
  }
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler" "api" {
  metadata {
    name      = "yosai-api-hpa"
    namespace = var.namespace
  }

  spec {
    min_replicas = 2
    max_replicas = 10

    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "yosai-api"
    }

    metric {
      type               = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }
  }
}

output "api_endpoint" {
  value = kubernetes_service.api.spec.0.cluster_ip
}