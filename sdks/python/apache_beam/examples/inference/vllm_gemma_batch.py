from __future__ import annotations

import faulthandler
faulthandler.enable()

import os
import logging
import tempfile
import asyncio
from collections.abc import Iterable
import uuid
import time
import random

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.metrics import Metrics # <-- 1. IMPORT METRICS

COMPLETION_EXAMPLES = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "John cena is",
]


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
        help=
        "GCS path to the directory containing model files (e.g., gs://bucket/models/gemma-2b-it/).",
    )

# 2. ADD A REUSABLE COUNTING DoFn
class CountFn(beam.DoFn):
    """A simple DoFn that counts elements and passes them through."""
    def __init__(self, namespace, name):
        super().__init__()
        self.counter = Metrics.counter(namespace, name)

    def process(self, element):
        self.counter.inc()
        yield element


# 3. UPDATE THE POST-PROCESSOR WITH COUNTERS
class GemmaPostProcessor(beam.DoFn):
  def __init__(self):
    super().__init__()
    # Counter for elements received by this DoFn
    self.received_counter = Metrics.counter('GemmaPostProcessor', 'elements_received')
    # Counter for elements that are successfully processed
    self.success_counter = Metrics.counter('GemmaPostProcessor', 'elements_succeeded')
    # Counter for elements that are empty or failed
    self.failure_counter = Metrics.counter('GemmaPostProcessor', 'elements_failed_or_empty')

  def process(self, element: PredictionResult):
    self.received_counter.inc()

    # We unpack it to get the original prompt. We don't need the key here.
    _, prompt = element.example

    vllm_output = element.inference

    # Check for empty or invalid output from the model
    if not vllm_output or not vllm_output.outputs:
      self.failure_counter.inc()
      logging.warning(f"Received empty or invalid inference for prompt: {prompt}")
      return

    choice = vllm_output.outputs[0]
    self.success_counter.inc() # Increment success counter before yielding
    yield {
        "prompt": prompt,
        "completion": choice.text.strip(),
        "prompt_tokens": len(vllm_output.prompt_token_ids),
        "completion_tokens": len(choice.token_ids),
    }


class VLLMModelHandlerGCS(ModelHandler[str, PredictionResult, object]):
  def __init__(self, model_gcs_path: str, vllm_kwargs: dict | None = None):
    logging.info(
        f"[MODEL HANDLER INIT] Initializing with model_gcs_path: {model_gcs_path}"
    )
    self._model_gcs_path = model_gcs_path
    self._vllm_kwargs = vllm_kwargs or {}
    self._local_path: str | None = None
    self._loop: asyncio.AbstractEventLoop | None = None

  def batch_elements_kwargs(self):
    # Use 128 as max_batch_size (or from vllm_kwargs if provided)
    return {"max_batch_size": self._vllm_kwargs.get("max_num_seqs", 128)}

  def check_gpu(self):
    import torch
    logging.info(
        "[MODEL HANDLER] Cuda Available: %s", torch.cuda.is_available())

  def load_model(self):
    # This is the new, hyper-detailed load_model method
    logging.info("--- [MODEL HANDLER] Starting granular load_model() ---")
    start_time = time.time()

    self.check_gpu() # This already logs GPU availability

    # --- Granular Import Block ---
    try:
        logging.info("[IMPORT_STEP 1/6 START] Importing vllm.engine.async_llm_engine")
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        logging.info("[IMPORT_STEP 1/6 SUCCESS] Imported vllm.engine.async_llm_engine")

        logging.info("[IMPORT_STEP 2/6 START] Importing vllm.engine.arg_utils")
        from vllm.engine.arg_utils import AsyncEngineArgs
        logging.info("[IMPORT_STEP 2/6 SUCCESS] Imported vllm.engine.arg_utils")

        logging.info("[IMPORT_STEP 3/6 START] Importing vllm.usage.usage_lib")
        from vllm.usage.usage_lib import UsageContext
        logging.info("[IMPORT_STEP 3/6 SUCCESS] Imported vllm.usage.usage_lib")
        
        logging.info("[IMPORT_STEP 4/6 SUCCESS] All vLLM libraries imported successfully.")

    except Exception as e:
        logging.error(f"[MODEL HANDLER] A standard Python exception occurred during import: {e}", exc_info=True)
        raise e
    # --- End Granular Import Block ---


    logging.info("[MODEL HANDLER] Now proceeding to download model from GCS...")
    if self._local_path is None:
      local_dir = tempfile.mkdtemp()
      self._local_path = local_dir
      gcs_dir = self._model_gcs_path.rstrip('/')
      pattern = f"{gcs_dir}/*"
      logging.info(f"[MODEL HANDLER] Matching GCS pattern: {pattern}")
      matches = FileSystems.match([pattern])
      metas = matches[0].metadata_list if matches else []
      if not metas:
        raise RuntimeError(f"No files found matching pattern {pattern}.")
      
      logging.info(f"[MODEL HANDLER] Found {len(metas)} files. Starting download to {local_dir}...")
      for i, m in enumerate(metas):
        src = m.path
        dst = os.path.join(local_dir, os.path.basename(src))
        logging.info(f"[MODEL HANDLER] Downloading file {i+1}/{len(metas)}: {os.path.basename(src)}")
        with FileSystems.open(src, 'rb') as fs, open(dst, 'wb') as fd:
          fd.write(fs.read())
      logging.info(f"[MODEL HANDLER] GCS download complete.")


    engine_args = {
        "model": self._local_path,
        "engine_use_ray": False,
        "enforce_eager": self._vllm_kwargs.get("enforce_eager", False),
        "gpu_memory_utilization": self._vllm_kwargs.get(
            "gpu_memory_utilization", 0.8),
        "dtype": self._vllm_kwargs.get("dtype", "float16"),
        "max_num_seqs": self._vllm_kwargs.get("max_num_seqs", 128),
    }
    args = AsyncEngineArgs(**engine_args)
    logging.info(
        f"[MODEL HANDLER] [IMPORT_STEP 5/6 START] Creating AsyncLLMEngine with args: {engine_args}")
    engine = AsyncLLMEngine.from_engine_args(
        args, usage_context=UsageContext.API_SERVER)
    logging.info("[MODEL HANDLER] [IMPORT_STEP 5/6 SUCCESS] AsyncLLMEngine created.")

    logging.info("[MODEL HANDLER] [IMPORT_STEP 6/6 START] Setting up asyncio event loop.")
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    logging.info("[MODEL HANDLER] [IMPORT_STEP 6/6 SUCCESS] Event loop set up.")

    elapsed = time.time() - start_time
    logging.info(
        f"--- [MODEL HANDLER] granular load_model() finished in {elapsed:.2f}s ---")
    return engine

  def run_inference(
      self,
      batch: list[str],
      model: object,
      inference_args: dict | None = None,
  ) -> Iterable[PredictionResult]:
    logging.info(
        f"--- [MODEL HANDLER] Starting run_inference() for batch size {len(batch)} ---"
    )
    start_time = time.time()

    if self._loop is None:
      self._loop = asyncio.new_event_loop()
      asyncio.set_event_loop(self._loop)

    from vllm import SamplingParams
    sampling_params = SamplingParams(max_tokens=1024)

    async def _get_final_output(prompt: str):
      req_id = str(uuid.uuid4())
      gen = model.generate(prompt, sampling_params, req_id)
      final_op = None
      async for op in gen:
        final_op = op
      return final_op

    async def _run_batch_async():
      tasks = [_get_final_output(prompt) for prompt in batch]
      return await asyncio.gather(*tasks, return_exceptions=True)

    outputs = self._loop.run_until_complete(_run_batch_async())
    results: list[PredictionResult] = []
    for example, inference in zip(batch, outputs):
      if isinstance(inference, Exception):
        logging.error(
            f"[MODEL HANDLER] Inference exception for prompt '{example}': {inference}"
        )
        # We still pass the exception on so the PostProcessor can count it as a failure
        results.append(PredictionResult(example, None))
        continue
      results.append(PredictionResult(example, inference))

    elapsed_inf = time.time() - start_time
    logging.info(
        f"--- [MODEL HANDLER] run_inference() finished in {elapsed_inf:.2f}s ---"
    )
    return results


def run(argv=None, save_main_session=True, test_pipeline=None):
  # Build pipeline options
  opts = PipelineOptions(argv)

  gem = opts.view_as(GemmaVLLMOptions)
  opts.view_as(SetupOptions).save_main_session = save_main_session

  logging.info(f"Pipeline starting with model path: {gem.model_gcs_path}")
  handler = VLLMModelHandlerGCS(
      model_gcs_path=gem.model_gcs_path,
      vllm_kwargs={
          "gpu_memory_utilization": 0.7,
          "dtype": "float16",
          "max_num_seqs": 128
      },
  )

  with (test_pipeline or beam.Pipeline(options=opts)) as p:
    # 4. ADD THE COUNTERS TO THE PIPELINE DEFINITION
    processed_elements = (
        p
        # | "ReadPrompts" >> beam.io.ReadFromText(gem.input_file)
        | "ReadPrompts" >> beam.Create(COMPLETION_EXAMPLES)
        | "CountRawReads" >> beam.ParDo(CountFn("pipeline", "prompts_read_from_file"))
        | "NonEmpty" >> beam.Filter(lambda l: l.strip())
        | "CountNonEmpty" >> beam.ParDo(CountFn("pipeline", "prompts_non_empty"))
        # Using 25 keys as an example, matching the max_num_workers.
        | "AddRandomKey" >> beam.Map(lambda x: (random.randint(0, 24), x))
        | "Infer" >> RunInference(handler)
        # PostProcessor now has its own internal counters
        | "Post" >> beam.ParDo(GemmaPostProcessor())
    )

    # Add a final counter before writing to BQ to see what is being loaded
    (
        processed_elements
        | "CountElementsForBQ" >> beam.ParDo(CountFn("pipeline", "elements_to_bq"))
        | "WriteToBQ" >> beam.io.WriteToBigQuery(
            gem.output_table,
            schema=
            "prompt:STRING,completion:STRING,prompt_tokens:INTEGER,completion_tokens:INTEGER",
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            method=beam.io.WriteToBigQuery.Method.FILE_LOADS,
        ))

  return p.result


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    logging.getLogger().setLevel(logging.INFO)
    run()