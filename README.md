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

1. Install `ADSK_AILAB_RAY` python package (in a new Python environment):

   ```python
   pip3 install --extra-index-url https://art-bobcat.autodesk.com/artifactory/api/pypi/team-gen-ai-accel-pypi/simple adsk-ailab-ray
   ```
2.  Login to AWS CLI using the following command:

```bash
python -m adsk_ailab_ray.tools.aws_cli_login --username YOUR_ADSK_USER_NAME # defaults to $USER
```

3. Create a Ray cluster by running the following command:

    ```python
    python3 -m adsk_ailab_ray.cluster.create --worker_node_types p3.16xlarge,p3dn.24xlarge --tag_value CADGPT
    ```
Adjust the `--worker_node_types` parameter as needed to specify the desired worker node types.

4. Submit a model training job using the provided command:

   ```bash
   ray job submit --address 'http://localhost:8265' --working-dir . --runtime-env-json='{"pip": "requirements_ray.txt"}' -- python train_ray.py --max_epochs 100 --num_gpus 16 --exp_name test_cadllm --dataset /home/ray/data --results_dir /home/ray/ray_results --strategy fsdp --model_name google/byt5-base
   ```

   Adjust the command parameters as needed. The `fsdp`, `deepspeed` and `ddp` strategies are supported.

5. Monitor the cluster and job status using the Ray dashboard. Access the dashboard by opening the following URL in your web browser:

    ```plaintext
    http://localhost:8265
    ```
    The dashboard provides real-time information about the cluster, including resource utilization, task and actor information, and logs. 

6. Monitor the TensorBoard logs by opening the following URL in your web browser:

    ```plaintext
    http://localhost:6006
    ```
7. Get an SSH session to the head node using the following command:

    ```python
    python3 -m adsk_ailab_ray.cluster.attach
    ```
    Note: The checkpoints will be automatically copied to the output S3 bucket at the end of training.

8. To cancel submitted jobs in Ray, you can use the following command:

    ```bash
    ray job stop --address 'http://localhost:8265' JOB_ID
    ```
    
    Replace `JOB_ID` with the actual ID of the job you want to cancel. You can find the job ID printed in the terminal when submitting the job, or you can locate it in the Ray dashboard.

    **Note:** If jobs are canceled before completion, the GPU memory may not be properly released. In such cases, you may need to remove and then recreate the cluster before submitting new jobs.

9. To remove the cluster and associated resources, run the following command:

    ```python
    python3 -m adsk_ailab_ray.cluster.remove
    ```
    This will terminate all the instances and delete the associated resources.

For more detailed instructions and additional information, please refer to the [Distributed ML Ray](https://git.autodesk.com/Research/distributed-ml-ray).

### Examples: Large Scale Distributed Training with Ray 

Create a cluster with P5 workers
```bash
python -m adsk_ailab_ray.cluster.create --worker_node_types p5.48xlarge --use_spot_workers --ebs_volume_size 300 --tag_value CADGPT
```

Submit a job to train `google/byt5-xl` (3.7 billion parameters) the cluster
```bash
   ray job submit --address 'http://localhost:8265' --working-dir . --runtime-env-json='{"pip": "requirements_ray.txt"}' -- python train_ray.py --max_epochs 1 --num_gpus 8 --exp_name test_byte5-xl --dataset /home/ray/data --results_dir /home/ray/ray_results --strategy fsdp --mix_precession --model_name google/byt5-xl
```
