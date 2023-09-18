FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS ka_digrec_runtime

COPY --from=ka_digrec_build /venv /venv

WORKDIR /mlflow/projects/code
COPY tools/run.py /mlflow/project/code/

ENTRYPOINT ["/venv/bin/python", "/mlflow/project/code/run.py"]
