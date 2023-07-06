# CAD LLM
Finetuning a Large Language Model (LLM) to autocomplete CAD engineering sketches.

## Setup
We use the [conda](https://www.anaconda.com/download/) distribution to manage dependencies. To setup the environment:

```
conda env create -f env.yml
conda activate cad_llm
```

## Data Prep
The `preprocess` directory contains scripts to convert and prepare the data for training. Run the scripts as modules, for example:

```
python -m preprocess.convert_obj_to_strings --input path/to/input/data --output path/to/output/data
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


### Distributed Training with Ray

1. Clone the [Distributed ML Ray](https://git.autodesk.com/Research/distributed-ml-ray) repository:

   ```bash
   git clone https://git.autodesk.com/Research/distributed-ml-ray
   cd distributed-ml-ray
   ```

2. Create a Ray cluster by running the following command:

    ```bash
    bash ray/scripts/create_cluster.sh --worker_node_types p4d.24xlarge
    ```
Adjust the `--worker_node_types` parameter as needed to specify the desired worker node types.

3. Submit a model training job using the provided command:

   ```bash
   RAY_ADDRESS='http://localhost:8265' ray job submit --working-dir path/to/cad_llm --runtime-env-json='{"pip": "requirements_ray.txt"}' -- python train_ray.py --max_epochs 100 --num_gpus 16 --exp_name test_cadllm --dataset /home/ray/data --results_dir /home/ray --strategy fsdp --model_name google/byt5-large
   ```

   Adjust the command parameters as needed.

4. Monitor the cluster and job status using the Ray dashboard. Access the dashboard by opening the following URL in your web browser:

    ```plaintext
    http://localhost:8265
    ```
    The dashboard provides real-time information about the cluster, including resource utilization, task and actor information, and logs. 

5. To cancel submitted jobs in Ray, you can use the following command:

    ```bash
    RAY_ADDRESS='http://localhost:8265' ray job stop JOB_ID
    ```
    
    Replace `JOB_ID` with the actual ID of the job you want to cancel. You can find the job ID printed in the terminal when submitting the job, or you can locate it in the Ray dashboard.

    **Note:** If jobs are canceled before completion, the GPU memory may not be properly released. In such cases, you may need to run the bash `ray/scripts/remove_cluster.sh` script to terminate all instances and delete associated resources, and then recreate the cluster using bash `ray/scripts/create_cluster.sh` before submitting new jobs.

6. To remove the cluster and associated resources, run the following command:

    ```bash
    bash ray/scripts/remove_cluster.sh
    ```
    This will terminate all the instances and delete the associated resources.

For more detailed instructions and additional information, please refer to the [Distributed ML Ray](https://git.autodesk.com/Research/distributed-ml-ray).

