#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Batch pipeline: Gemma-2B-it via vLLM Async Engine → BigQuery, loading model from GCS.
"""

from __future__ import annotations

import os
import gc
import sys
import logging
import tempfile
import multiprocessing as mp
import asyncio
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

# vLLM async engine imports
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext

# Force safe multiprocessing start method early (avoid CUDA/fork issues).
mp.set_start_method("spawn", force=True)

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
    def process(self, element: PredictionResult):
        prompt = element.example
        vllm_output = element.inference

        choice = vllm_output.outputs[0]
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": len(choice.token_ids),
        }

# =================================================================
# 3. Model Handler for loading from GCS with AsyncLLMEngine
# =================================================================
class VLLMModelHandlerGCS(ModelHandler[str, PredictionResult, object]):
    def __init__(self, model_gcs_path: str, vllm_kwargs: dict | None = None):
        self._model_gcs_path = model_gcs_path
        self._vllm_kwargs = vllm_kwargs or {}
        self._local_path: str | None = None  # cached after download
        self._loop: asyncio.AbstractEventLoop | None = None

    def batch_elements_kwargs(self):
        # Align max_batch_size with max_num_seqs
        return {"max_batch_size": self._vllm_kwargs.get("max_num_seqs", 16)}

    def load_model(self):
        # Download model artifacts from GCS once per worker
        if self._local_path is None:
            local_model_dir = tempfile.mkdtemp()
            self._local_path = local_model_dir
            gcs_dir = self._model_gcs_path.rstrip('/')
            pattern = f"{gcs_dir}/*"

            match_results = FileSystems.match([pattern])
            metadata_list = match_results[0].metadata_list if match_results else []
            if not metadata_list:
                raise RuntimeError(f"No files found matching pattern {pattern}.")

            for metadata in metadata_list:
                src = metadata.path
                dst = os.path.join(local_model_dir, os.path.basename(src))
                with FileSystems.open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                    f_dst.write(f_src.read())

        # Build engine arguments from kwargs
        engine_args = AsyncEngineArgs(
            model=self._local_path,
            engine_use_ray=False,
            enforce_eager=self._vllm_kwargs.get("enforce_eager", False),
            gpu_memory_utilization=self._vllm_kwargs.get("gpu_memory_utilization", 0.8),
            dtype=self._vllm_kwargs.get("dtype", "bfloat16"),
            max_num_seqs=self._vllm_kwargs.get("max_num_seqs", 16),
        )
        engine = AsyncLLMEngine.from_engine_args(
            engine_args,
            usage_context=UsageContext.API_SERVER,
        )

        # Create and store a dedicated event loop for async execution
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return engine

    def run_inference(
        self,
        batch: list[str],
        model: AsyncLLMEngine,
        inference_args: dict | None = None,
    ) -> Iterable[PredictionResult]:
        logging.info(f"Running async inference on batch of size {len(batch)}")
        # Schedule the async generate() call on our event loop
        coroutine = model.generate(batch, **(inference_args or {}))
        results = self._loop.run_until_complete(coroutine)
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
        vllm_kwargs={
            "gpu_memory_utilization": 0.8,
            "dtype": "bfloat16",
            "max_num_seqs": 16,
        },
    )

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
