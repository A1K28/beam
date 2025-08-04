# Licensed to the Apache Software Foundation (ASF) ...
"""
Batch pipeline: Gemma-2B-it via vLLM → BigQuery, loading model from GCS.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)


class GemmaPostProcessor(beam.DoFn):
    def process(self, element: tuple[str, PredictionResult]) -> Iterable[dict]:
        prompt, pred = element
        # The vLLM output is stored in the `inference` field of PredictionResult
        vllm_output = pred.inference
        choice = vllm_output.outputs[0]
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": len(choice.token_ids),
        }


class VLLMModelHandlerGCS(ModelHandler[str, PredictionResult, object]):
    """
    A ModelHandler that downloads model artifacts from Google Cloud Storage
    and loads them into a vLLM engine.
    """

    def __init__(self, model_gcs_path: str, vllm_kwargs: dict | None = None):
        self._model_gcs_path = model_gcs_path
        self._vllm_kwargs = vllm_kwargs or {}

    def load_model(self):
        """
        This method is called on the Dataflow worker. It downloads all files
        from the GCS path to a temporary local directory, then loads the model
        using vLLM.
        """
        from vllm import LLM

        # Create a temporary local directory to hold the model files
        local_model_dir = tempfile.mkdtemp()

        # Use Beam's FileSystems to find and copy all model files
        gcs_path_pattern = FileSystems.join(self._model_gcs_path, "*")
        file_metadata_list = FileSystems.match([gcs_path_pattern])[0].metadata_list

        logging.info(f"Starting model download from {self._model_gcs_path} to {local_model_dir}")
        for metadata in file_metadata_list:
            source_path = metadata.path
            destination_path = FileSystems.join(local_model_dir, FileSystems.split(source_path)[-1])
            FileSystems.copy([source_path], [destination_path])
        logging.info("Model download complete.")

        # Initialize the vLLM engine with the local path
        return LLM(model=local_model_dir, **self._vllm_kwargs)

    def run_inference(
        self, batch: list[str], model: object, inference_args: dict | None = None
    ) -> Iterable[PredictionResult]:
        """
        Runs inference on a batch of prompts.
        """
        # The 'model' object is the vLLM engine returned by load_model()
        results = model.generate(batch, **(inference_args or {}))
        return [PredictionResult(example, result) for example, result in zip(batch, results)]


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", "--input_file", dest="input", required=True,
        help="gs://bucket/prompts.txt (one prompt per line)",
    )
    p.add_argument(
        "--output_table", required=True,
        help="project:dataset.gemma_batch",
    )
    p.add_argument(
        "--model_gcs_path", required=True,
        help="GCS path to the directory containing model files (e.g., gs://bucket/models/gemma-2b-it/)",
    )
    # Perf-test extras (accepted but unused here)
    p.add_argument("--publish_to_big_query", default=False)
    p.add_argument("--metrics_dataset")
    p.add_argument("--metrics_table")
    p.add_argument("--influx_measurement")
    p.add_argument("--device", default="GPU")
    return p.parse_known_args(argv)


def run(argv=None, save_main_session=True, test_pipeline=None):
    opts, pipeline_args = _parse(argv)

    # Use the new handler that loads from GCS
    handler = VLLMModelHandlerGCS(
        model_gcs_path=opts.model_gcs_path,
        vllm_kwargs={"gpu_memory_utilization": 0.9, "dtype": "bfloat16"},
    )

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = False
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with (test_pipeline or beam.Pipeline(options=pipeline_options)) as p:
        (
            p
            | "ReadPrompts" >> beam.io.ReadFromText(opts.input)
            | "StripBlank" >> beam.Filter(lambda ln: ln.strip())
            | "RunInference" >> RunInference(handler)
            | "PostProcess" >> beam.ParDo(GemmaPostProcessor())
            | "WriteBQ"
            >> beam.io.WriteToBigQuery(
                opts.output_table,
                schema=(
                    "prompt:STRING, completion:STRING,"
                    "prompt_tokens:INTEGER, completion_tokens:INTEGER"
                ),
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                method=beam.io.WriteToBigQuery.Method.FILE_LOADS,
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()