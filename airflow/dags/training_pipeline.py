from __future__ import annotations 
from airflow import DAG
from airflow.operators.python import PythonOperator
from logger.logging import logging
from pipelines.training_pipeline import TrainingPipeline
import pendulum
from textwrap import dedent
import json

Training_obj = TrainingPipeline()

with DAG(
    "Diamond_price_prediction",
    default_args={"retries": 2},
    description = "Training pipeline",
    schedule = "@weekly",
    start_date= pendulum.datetime(2024, 7, 12, tz="UTC"),
    catchup=False,
    tags=["machine_learning", "price", "prediction"],
) as dag:
    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        trainpath, testpath = Training_obj.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":trainpath,"test_data_path":testpath})


    def data_transformtion(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifacts")
        train_data, test_data = Training_obj.start_data_transformtion( data_ingestion_artifact["train_data_path"],data_ingestion_artifact["test_data_path"])
        train_data = train_data.tolist()
        test_data  = test_data .tolist()
        ti.xcom_push("data_transformation_artifact", {"training_data": train_data, "testing_data": test_data})

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        train_arr = np.array(data_transformation_artifact["train_data"])
        test_arr = np.array(data_transformation_artifact["test_data"])
        Training_obj.start_model_training(train_arr, test_arr)

    def push_data_to_azureblob(**kwargs):
        import os
        bucket_name="reposatiory_name"
        artifact_folder="/app/artifacts"
        #you can save it ti the azure blob
        #os.system(f"aws s3 sync {artifact_folder} s3:/{bucket_name}/artifact")


    data_ingestion_task =PythonOperator(
        task_id="data_ingestion",
        python_callable = data_ingestion
    )
    data_ingestion_task.doc_md = dedent(
    """\
        #### Ingestion task
        this task creates a train and test file.
        """
    )

    data_transformtion_task = PythonOperator(
        task_id = "data_transformation",
        python_callable = data_transformtion
    )

    data_transformtion_task.doc_md = dedent(
    """\
        #### Transformation task
        this task performs the transformation
        """
    )

    model_trainer_task = PythonOperator(
        task_id = "model_trainer",
        python_callable = model_trainer
    )
    model_trainer_task.doc_md = dedent(
    """\
        #### model trainer task
        this task perform training
        """
    )

    push_to_s3_task = PythonOperator(
        task_id = "push_data_to_s3",
        python_callable =  push_data_to_azureblob
    )


data_ingestion_task >> data_transformtion_task >> model_trainer_task >> push_to_s3_task