# Object Recognition 
This code provides the foundation for creating a Neural Network based on InceptionV3 to recognize objects in images

## Requirements
- Python version 3.11.5
- virtualenv

The script is configured to work with two default directories for the images dataset:
- test
- train

To use the script without modifications, create these directories at the same level as the main script and place the images into them.

> [!IMPORTANT]
> The script can detect NVIDIA GPUs and utilize them for the training process if you have the NVIDIA driver, CUDA Toolkit, and CUDNN drivers installed.

### Setting environment
After cloning the repository, open a terminal in the folder and execute the following commands:
```
virtualenv -p 3.11 venv
. venv/scripts/activate
python -m pip install -r requirements.txt
```

These commands will install the necessary dependencies. Now, you can run the program as a usual Python script.
