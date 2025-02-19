#!/bin/bash

# Set the namespace
NAMESPACE="mlflow"

# Check if values.yaml exists
if [[ ! -f values.yaml ]]; then
    echo "Error: values.yaml file not found!"
    exit 1
fi

# Add the Bitnami repository if it's not already added
helm repo list | grep -q 'bitnami' || helm repo add bitnami https://charts.bitnami.com/bitnami

# Update the Helm repositories
helm repo update

# Run the Helm install command from the Bitnami repository
helm upgrade --install mlflow bitnami/mlflow -f values.yaml --create-namespace -n $NAMESPACE

# Check if Helm install was successful
if [[ $? -eq 0 ]]; then
    echo "MLflow installed successfully in the '$NAMESPACE' namespace."
else
    echo "Helm installation failed."
    exit 1
fi

echo "Please run the following command to port-forward the MLflow UI to your local machine:"
echo "kubectl port-forward deployment/mlflow-tracking 5000:5000 -n $NAMESPACE"

