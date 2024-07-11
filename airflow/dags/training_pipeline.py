from __future__ import annotations 
from airflow import Dag
from airflow.operators.python import PythonOperator
from pipeline.training_pipeline import TrainingPipeline
import pendulum
from textwrap import dedent
import json

Training_obj = TrainingPipeline()

with Dag(
    "Diamond_price_prediction",
    default_args("retries": 2),
    description = " Training pipeline",
    schedule = "@weekly",
    start_data = pendulum.datetime(2024, 7, 12, tz="UTC")
    catchup=False,
    tags=["machine_learning", "price", "prediction"],
) as dag:
    dag.doc_md = __doc__

