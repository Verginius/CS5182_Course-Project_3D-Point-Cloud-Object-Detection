#!/bin/bash
# Activate pointcloud_5090 conda environment for this project

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pointcloud_5090" ]; then
    # Check if environment exists
    if conda env list | grep -q "^pointcloud_5090 "; then
        echo "Activating conda environment: pointcloud_5090"
        eval "$(conda shell.bash hook)"
        conda activate pointcloud_5090
    else
        echo "Environment 'pointcloud_5090' not found."
        echo "To create it, run:"
        echo "  conda env create -f environment.yml"
        echo "  conda activate pointcloud_5090"
    fi
fi
