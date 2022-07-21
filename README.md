# Image Classification using AWS SageMaker

AWS Sagemaker is used to to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

The objective of the project is not to create the best trained model, but rather demonstrate the techniques and tools that can be used to create, profile, debug, deploy and use a model using the AWS SageMaker infrastructure. In that context the specific neural network chosen is not important.

I have also taken the opportunity to experiment with pytorch's new datapipes when accessing training and test data. 

SageMaker's Debugger has been used to instrument training so that any issues can be more easily identified and corrected.

The images used are different dog breeds, like the ones below.

dog-breeds-sample.png

The project uses PyTorch to perform the ML operations.

## Project Set Up and Installation
You can either use this project in conjunction wuth SageMaker Studio or run the notebook locally just utilising AWS infrastructure as required.

If using SageMaker Studio log into AWS and go to the SageMaker Studio. There you can clone this repository and run the train_and_deploy.ipynb notebook.

If you choose to run the notebook locally you will need to install conda/miniconda and install the required packages. I would recommend using miniconda (https://docs.conda.io/en/latest/miniconda.html). The required packages are included in the notebook.


## Dataset
The dataset used is that of dog breeds. There are separate folders for train, validate and test. In each of these folders there are separate folders for the different dog breeds. Images organised in this structure, with the labels effectively being the folder names is supported in PyTorch by the ImageFolder (https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) which makes handling this data structure straight-forward.

The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

Used PyTorch Datapipes in order to determine means and standard deviations of the images.
Able to deploy updated script via model


