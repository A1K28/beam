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
import uuid
import time

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

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

        if not vllm_output or not vllm_output.outputs:
            return

        choice = vllm_output.outputs[0]
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": len(vllm_output.prompt_token_ids),
            "completion_tokens": len(choice.token_ids),
        }

# =================================================================
# 3. Model Handler for loading from GCS with AsyncLLMEngine (lazy imports)
# =================================================================
class VLLMModelHandlerGCS(ModelHandler[str, PredictionResult, object]):
    def __init__(self, model_gcs_path: str, vllm_kwargs: dict | None = None):
        logging.info(f"[MODEL HANDLER INIT] Initializing with model_gcs_path: {model_gcs_path}")
        self._model_gcs_path = model_gcs_path
        self._vllm_kwargs = vllm_kwargs or {}
        self._local_path: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def batch_elements_kwargs(self):
        # Use 128 as max_batch_size (or from vllm_kwargs if provided)
        return {"max_batch_size": self._vllm_kwargs.get("max_num_seqs", 128)}

    def check_gpu(self):
        import torch
        logging.info("[MODEL HANDLER] Cuda Available: %s", torch.cuda.is_available())

    def load_model(self):
        logging.info("--- [MODEL HANDLER] Starting load_model() ---")
        start_time = time.time()

        self.check_gpu()
        logging.info("[MODEL HANDLER] Importing vLLM libraries...")
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        logging.info("[MODEL HANDLER] vLLM libraries imported successfully.")

        if self._local_path is None:
            local_dir = tempfile.mkdtemp()
            self._local_path = local_dir
            gcs_dir = self._model_gcs_path.rstrip('/')
            pattern = f"{gcs_dir}/*"
            matches = FileSystems.match([pattern])
            metas = matches[0].metadata_list if matches else []
            if not metas:
                raise RuntimeError(f"No files found matching pattern {pattern}.")
            for m in metas:
                src = m.path
                dst = os.path.join(local_dir, os.path.basename(src))
                with FileSystems.open(src, 'rb') as fs, open(dst, 'wb') as fd:
                    fd.write(fs.read())

        engine_args = {
            "model": self._local_path,
            "engine_use_ray": False,
            "enforce_eager": self._vllm_kwargs.get("enforce_eager", False),
            "gpu_memory_utilization": self._vllm_kwargs.get("gpu_memory_utilization", 0.8),
            "dtype": self._vllm_kwargs.get("dtype", "bfloat16"),
            "max_num_seqs": self._vllm_kwargs.get("max_num_seqs", 128),
        }
        args = AsyncEngineArgs(**engine_args)
        logging.info("[MODEL HANDLER] Creating AsyncLLMEngine (this can take minutes)...")
        engine = AsyncLLMEngine.from_engine_args(args, usage_context=UsageContext.API_SERVER)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        elapsed = time.time() - start_time
        logging.info(f"--- [MODEL HANDLER] load_model() finished in {elapsed:.2f}s ---")
        return engine

    def run_inference(
        self,
        batch: list[str],
        model: object,
        inference_args: dict | None = None,
    ) -> Iterable[PredictionResult]:
        logging.info(f"--- [MODEL HANDLER] Starting run_inference() for batch size {len(batch)} ---")
        start_time = time.time()

        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=1024, max_num_seqs=self._vllm_kwargs.get("max_num_seqs", 128))

        async def _run_batch_async():
            tasks = [model.generate(prompt, sampling_params, str(uuid.uuid4())) for prompt in batch]
            return await asyncio.gather(*tasks, return_exceptions=True)

        outputs = self._loop.run_until_complete(_run_batch_async())
        results: list[PredictionResult] = []
        for example, inference in zip(batch, outputs):
            if isinstance(inference, Exception):
                continue
            results.append(PredictionResult(example, inference))

        logging.info(f"--- [MODEL HANDLER] run_inference() finished in {time.time() - start_time:.2f}s ---")
        return results

# =================================================================
# 4. Pipeline Execution
# =================================================================
from apache_beam.options.pipeline_options import GoogleCloudOptions

def run(argv=None, save_main_session=True, test_pipeline=None):
    # Build pipeline options
    opts = PipelineOptions(argv)

    # Disable dynamic source splitting (work rebalancing) to prevent empty-split errors
    gcloud_opts = opts.view_as(GoogleCloudOptions)
    gcloud_opts.experiments = gcloud_opts.experiments or []
    if 'disable_work_rebalancing' not in gcloud_opts.experiments:
        gcloud_opts.experiments.append('disable_work_rebalancing')

    gem = opts.view_as(GemmaVLLMOptions)
    opts.view_as(SetupOptions).save_main_session = save_main_session

    logging.info(f"Pipeline starting with model path: {gem.model_gcs_path}")
    handler = VLLMModelHandlerGCS(
        model_gcs_path=gem.model_gcs_path,
        vllm_kwargs={"gpu_memory_utilization": 0.8, "dtype": "bfloat16", "max_num_seqs": 128},
    )

    with (test_pipeline or beam.Pipeline(options=opts)) as p:
        (
            p
            | "ReadPrompts" >> beam.io.ReadFromText(gem.input_file)
            | "NonEmpty" >> beam.Filter(lambda l: l.strip())
            | "Infer" >> RunInference(handler)
            | "Post" >> beam.ParDo(GemmaPostProcessor())
            | "WriteToBQ" >> beam.io.WriteToBigQuery(
                gem.output_table,
                schema="prompt:STRING,completion:STRING,prompt_tokens:INT,completion_tokens:INT",
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                method=beam.io.WriteToBigQuery.Method.FILE_LOADS,
            )
        )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
