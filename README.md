# CAD LLM
Finetuning a Large Language Model (LLM) to autocomplete CAD engineering sketches.

## Setup
We use the [conda](https://www.anaconda.com/download/) distribution to manage dependencies. To setup the environment:

```
conda env create -f env.yml
conda activate cad_llm
```

## Data Prep
The `preprocess` directory contains scripts to convert and prepare the data ready for training. Run the scripts as modules, for example:

```
python -m preprocess.convert_obj_to_strings --input path/to/input/data
```

## Training
The model can be trained both locally and using SageMaker training on AWS.

### Local Training
To run local training:
```
python main.py --exp_name my_experiment
```

To see a full list of available arguments:
```
python main.py --help
```

### SageMaker Training
Sagemaker training requires the additional packages:
- `sagemaker`
- `boto3`

To run SageMaker training:
```
python launch.py --aws_name my_name
```

To see a full list of available arguments:
```
python launch.py --help
```

