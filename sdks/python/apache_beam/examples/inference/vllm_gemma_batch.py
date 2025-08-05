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
        # LOGGING: Added to see the data structure post-inference
        logging.info(f"[POST-PROCESSOR] Received element: {element.example}")
        prompt = element.example
        vllm_output = element.inference

        if not vllm_output or not vllm_output.outputs:
            logging.error(f"[POST-PROCESSOR] Inference for prompt '{prompt}' returned empty or invalid output.")
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
        # LOGGING: Log constructor arguments
        logging.info(f"[MODEL HANDLER INIT] Initializing with model_gcs_path: {model_gcs_path}")
        self._model_gcs_path = model_gcs_path
        self._vllm_kwargs = vllm_kwargs or {}
        self._local_path: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def batch_elements_kwargs(self):
        return {"max_batch_size": self._vllm_kwargs.get("max_num_seqs", 16)}

    def load_model(self):
        # LOGGING: Announce start of the critical model loading process
        logging.info("--- [MODEL HANDLER] Starting load_model() ---")
        start_time = time.time()

        # LOGGING: Lazy import vLLM
        logging.info("[MODEL HANDLER] Importing vLLM libraries...")
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        logging.info("[MODEL HANDLER] vLLM libraries imported successfully.")

        if self._local_path is None:
            # LOGGING: Detail the GCS download process
            logging.info("[MODEL HANDLER] No local model path cached. Starting GCS download.")
            local_dir = tempfile.mkdtemp()
            self._local_path = local_dir
            logging.info(f"[MODEL HANDLER] Created temporary local directory: {self._local_path}")

            gcs_dir = self._model_gcs_path.rstrip('/')
            pattern = f"{gcs_dir}/*"
            logging.info(f"[MODEL HANDLER] Matching GCS files with pattern: {pattern}")

            matches = FileSystems.match([pattern])
            metas = matches[0].metadata_list if matches else []

            logging.info(f"[MODEL HANDLER] Found {len(metas)} model files to download.")
            if not metas:
                logging.error(f"CRITICAL: No files found matching GCS pattern {pattern}. This is a fatal error.")
                raise RuntimeError(f"No files found matching pattern {pattern}.")

            for i, m in enumerate(metas):
                src = m.path
                dst = os.path.join(local_dir, os.path.basename(src))
                logging.info(f"[MODEL HANDLER] Downloading file {i+1}/{len(metas)}: {src} to {dst}")
                try:
                    with FileSystems.open(src, 'rb') as fs, open(dst, 'wb') as fd:
                        fd.write(fs.read())
                except Exception as e:
                    logging.error(f"CRITICAL: Failed to download file {src}. Error: {e}", exc_info=True)
                    raise e
            logging.info("[MODEL HANDLER] All model files downloaded successfully.")

        # LOGGING: Detail the vLLM engine arguments
        logging.info("[MODEL HANDLER] Building AsyncEngineArgs...")
        engine_args_dict = {
            "model": self._local_path,
            "engine_use_ray": False,
            "enforce_eager": self._vllm_kwargs.get("enforce_eager", False),
            "gpu_memory_utilization": self._vllm_kwargs.get("gpu_memory_utilization", 0.8),
            "dtype": self._vllm_kwargs.get("dtype", "bfloat16"),
            "max_num_seqs": self._vllm_kwargs.get("max_num_seqs", 16),
        }
        logging.info(f"[MODEL HANDLER] vLLM Engine Args: {engine_args_dict}")
        args = AsyncEngineArgs(**engine_args_dict)

        # LOGGING: Announce engine creation (the slowest part)
        logging.info("[MODEL HANDLER] Creating AsyncLLMEngine from args. THIS MAY TAKE SEVERAL MINUTES...")
        try:
            engine = AsyncLLMEngine.from_engine_args(args, usage_context=UsageContext.API_SERVER)
            logging.info("[MODEL HANDLER] AsyncLLMEngine created successfully.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to create AsyncLLMEngine. Error: {e}", exc_info=True)
            raise e

        logging.info("[MODEL HANDLER] Creating new asyncio event loop.")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        end_time = time.time()
        logging.info(f"--- [MODEL HANDLER] load_model() finished successfully in {end_time - start_time:.2f} seconds. ---")
        return engine

    def run_inference(
        self,
        batch: list[str],
        model: object,
        inference_args: dict | None = None,
    ) -> Iterable[PredictionResult]:
        logging.info(f"--- [MODEL HANDLER] Starting run_inference() for batch of size {len(batch)} ---")
        start_time = time.time()
        
        if self._loop is None:
            logging.warning("[MODEL HANDLER] Event loop was None. Creating a new one for run_inference.")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        from vllm import SamplingParams
        logging.info("[MODEL HANDLER] Imported SamplingParams.")
        sampling_params = SamplingParams(max_tokens=1024)

        async def _get_final_output(prompt: str):
            request_id = str(uuid.uuid4())
            results_generator = model.generate(prompt, sampling_params, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            return final_output

        async def _run_batch_async():
            logging.info(f"[MODEL HANDLER] Creating {len(batch)} concurrent inference tasks.")
            tasks = [_get_final_output(prompt) for prompt in batch]
            logging.info("[MODEL HANDLER] Gathering results from tasks...")
            all_outputs = await asyncio.gather(*tasks, return_exceptions=True)
            logging.info("[MODEL HANDLER] All inference tasks completed.")
            return all_outputs

        outputs = self._loop.run_until_complete(_run_batch_async())
        
        results = []
        for example, inference in zip(batch, outputs):
            if isinstance(inference, Exception):
                logging.error(f"[MODEL HANDLER] Inference for prompt '{example}' failed with exception: {inference}", exc_info=inference)
                # Optionally yield a failure record or just skip
                continue
            results.append(PredictionResult(example, inference))
        
        end_time = time.time()
        logging.info(f"--- [MODEL HANDLER] run_inference() finished in {end_time - start_time:.2f} seconds. Yielding {len(results)} results. ---")
        return results

# =================================================================
# 4. Pipeline Execution
# =================================================================
def run(argv=None, save_main_session=True, test_pipeline=None):
    opts = PipelineOptions(argv)
    gem = opts.view_as(GemmaVLLMOptions)
    opts.view_as(SetupOptions).save_main_session = save_main_session

    logging.info(f"Pipeline starting with model path: {gem.model_gcs_path}")
    logging.info(f"Pipeline reading from: {gem.input_file}")
    logging.info(f"Pipeline writing to: {gem.output_table}")

    handler = VLLMModelHandlerGCS(
        model_gcs_path=gem.model_gcs_path,
        vllm_kwargs={"gpu_memory_utilization":0.8, "dtype":"bfloat16", "max_num_seqs":16},
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
    # This sets the root logger, which is what Beam workers will use.
    logging.getLogger().setLevel(logging.INFO)
    logging.info("--- Starting main execution ---")
    run()