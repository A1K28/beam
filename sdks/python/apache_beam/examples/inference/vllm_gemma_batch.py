# Licensed to the Apache Software Foundation (ASF) ...
"""
Batch pipeline: Gemma-2B-it via vLLM → BigQuery, loading model from GCS.
"""

from __future__ import annotations

import os
import logging
import tempfile
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

# =================================================================
# 1. Custom Pipeline Options
# =================================================================
class GemmaVLLMOptions(PipelineOptions):
    """Custom pipeline options for the Gemma vLLM batch inference job."""
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input",
            dest="input_file",
            required=True,
            help="Input file gs://path containing prompts.",
        )
        parser.add_argument(
            "--output_table",
            required=True,
            help="BigQuery table to write to in the form project:dataset.table.",
        )
        parser.add_argument(
            "--model_gcs_path",
            required=True,
            help="GCS path to the directory containing model files (e.g., gs://bucket/models/gemma-2b-it/).",
        )

# =================================================================
# 2. Post-processing DoFn
# =================================================================
class GemmaPostProcessor(beam.DoFn):
    def process(self, element: tuple[str, PredictionResult]) -> Iterable[dict]:
        prompt, pred = element
        vllm_output = pred.inference
        choice = vllm_output.outputs[0]
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": len(choice.token_ids),
        }

# =================================================================
# 3. Model Handler for loading from GCS
# =================================================================
class VLLMModelHandlerGCS(ModelHandler[str, PredictionResult, object]):
    """
    A ModelHandler that downloads model artifacts from Google Cloud Storage
    and loads them into a vLLM engine.
    """
    def __init__(self, model_gcs_path: str, vllm_kwargs: dict | None = None):
        self._model_gcs_path = model_gcs_path
        self._vllm_kwargs = vllm_kwargs or {}

    def load_model(self):
        from vllm import LLM
        local_model_dir = tempfile.mkdtemp()
        gcs_dir_path = self._model_gcs_path.rstrip('/')
        gcs_path_pattern = f"{gcs_dir_path}/*"

        logging.info(f"Searching for model files in GCS with pattern: {gcs_path_pattern}")

        match_results = FileSystems.match([gcs_path_pattern])
        if not match_results or not match_results[0].metadata_list:
            raise RuntimeError(f"No files found matching pattern {gcs_path_pattern}.")

        file_metadata_list = match_results[0].metadata_list
        logging.info(f"Found {len(file_metadata_list)} model files to download.")

        for metadata in file_metadata_list:
            source_path = metadata.path
            destination_filename = os.path.basename(source_path)
            destination_path = os.path.join(local_model_dir, destination_filename)
            logging.info(f"Copying {source_path} to {destination_path}...")
            with FileSystems.open(source_path, 'rb') as f_source:
                with open(destination_path, 'wb') as f_dest:
                    f_dest.write(f_source.read())

        logging.info(f"Model download complete. Contents of {local_model_dir}: {os.listdir(local_model_dir)}")
        return LLM(model=local_model_dir, **self._vllm_kwargs)

    def run_inference(
        self, batch: list[str], model: object, inference_args: dict | None = None
    ) -> Iterable[PredictionResult]:
        results = model.generate(batch, **(inference_args or {}))
        return [PredictionResult(example, result) for example, result in zip(batch, results)]

# =================================================================
# 4. Pipeline Execution
# =================================================================
def run(argv=None, save_main_session=True, test_pipeline=None):
    pipeline_options = PipelineOptions(argv)
    gemma_options = pipeline_options.view_as(GemmaVLLMOptions)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    handler = VLLMModelHandlerGCS(
        model_gcs_path=gemma_options.model_gcs_path,
        vllm_kwargs={"gpu_memory_utilization": 0.9, "dtype": "bfloat16"},
    )

    # This line is the key change: it uses the test_pipeline if it's
    # provided, otherwise it creates a new one.
    with (test_pipeline or beam.Pipeline(options=pipeline_options)) as p:
        (
            p
            | "ReadPrompts" >> beam.io.ReadFromText(gemma_options.input_file)
            | "StripBlank" >> beam.Filter(lambda ln: ln.strip())
            | "RunInference" >> RunInference(handler)
            | "PostProcess" >> beam.ParDo(GemmaPostProcessor())
            | "WriteBQ"
            >> beam.io.WriteToBigQuery(
                gemma_options.output_table,
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