import os
from pathlib import Path


list_of_files = [
".github/workflows/.gitkeep",
"src/components/__init__.py",
"src/components/data_ingestion.py",
"src/components/data_transformtion.py",
"src/components/model_trainer.py",
"src/components/model_evaluation.py",
"src/pipeline/__init__.py",
"src/pipeline/training_pipeline.py",
"src/pipeline/prediction_pipeline.py",
"src/utils/__init__.py",
"src/utils/utils.py",
"src/logger/logging.py",
"src/exception/exception.py",
"tests/unit/__init__.py",
"tests/unit/unit.py",
"tests/integration/integration.py",
"pyproject.toml",
"setup.py",
"setup.cfg",
"init_setup.sh",
"requirements.txt",
"requirements_dev.txt",
"experiment/experiment.ipynb",
"tox.ini"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, filename = os.path.split(filepath)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok = True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
