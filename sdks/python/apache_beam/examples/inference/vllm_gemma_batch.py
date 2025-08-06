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
from apache_beam.metrics import Metrics

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


class CountFn(beam.DoFn):
    """A simple DoFn that counts elements and passes them through."""
    def __init__(self, namespace, name):
        super().__init__()
        self.counter = Metrics.counter(namespace, name)

    def process(self, element):
        self.counter.inc()
        yield element


class GemmaPostProcessor(beam.DoFn):
  def __init__(self):
    super().__init__()
    self.received_counter = Metrics.counter('GemmaPostProcessor', 'elements_received')
    self.success_counter = Metrics.counter('GemmaPostProcessor', 'elements_succeeded')
    self.failure_counter = Metrics.counter('GemmaPostProcessor', 'elements_failed_or_empty')

  def process(self, element: PredictionResult):
    self.received_counter.inc()
    prompt, vllm_output = element.example, element.inference

    if not vllm_output or not vllm_output.outputs:
      self.failure_counter.inc()
      logging.warning(f"Received empty or invalid inference for prompt: {prompt}")
      return

    choice = vllm_output.outputs[0]
    self.success_counter.inc()
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

  def batch_elements_kwargs(self):
    return {"max_batch_size": self._vllm_kwargs.get("max_num_seqs", 128)}

  def check_gpu(self):
    import torch
    logging.info(
        "[MODEL HANDLER] Cuda Available: %s", torch.cuda.is_available())

  def load_model(self):
    """
    Loads model from GCS and initializes it using vLLM's high-level API.
    This does NOT use AsyncLLMEngine or AsyncEngineArgs.
    """
    logging.info("--- [MODEL HANDLER] Starting load_model() ---")
    start_time = time.time()

    self.check_gpu()

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
            with FileSystems.open(src, 'rb') as fs, open(dst, 'wb') as fd:
                fd.write(fs.read())
        logging.info(f"[MODEL HANDLER] GCS download complete.")

    # Use the high-level vLLM API. This avoids the problematic imports.
    try:
        from vllm import LLM
    except ImportError as e:
        logging.error(f"[MODEL HANDLER] Failed to import vLLM. Make sure vLLM is installed in the worker environment. Error: {e}", exc_info=True)
        raise e

    logging.info(f"[MODEL HANDLER] Creating vllm.LLM with args: {self._vllm_kwargs}")
    model = LLM(model=self._local_path, **self._vllm_kwargs)
    
    elapsed = time.time() - start_time
    logging.info(f"--- [MODEL HANDLER] load_model() finished in {elapsed:.2f}s ---")
    return model

  def run_inference(
      self,
      batch: list[str],
      model: object, # This will be an instance of vllm.LLM
      inference_args: dict | None = None,
  ) -> Iterable[PredictionResult]:
    """
    Runs inference on a batch of prompts using the vllm.LLM.generate method.
    This is much simpler as the LLM class handles batching internally.
    """
    logging.info(
        f"--- [MODEL HANDLER] Starting run_inference() for batch size {len(batch)} ---"
    )
    start_time = time.time()

    # Import SamplingParams here, as it's part of the vLLM API.
    from vllm import SamplingParams
    sampling_params = SamplingParams(max_tokens=1024)

    # The LLM class's generate method can take a list of prompts directly.
    outputs = model.generate(batch, sampling_params)

    # The output is a list of RequestOutput objects, in the same order as the input batch.
    elapsed_inf = time.time() - start_time
    logging.info(
        f"--- [MODEL HANDLER] vLLM processing finished in {elapsed_inf:.2f}s ---"
    )
    
    return [PredictionResult(example, inference) for example, inference in zip(batch, outputs)]


def run(argv=None, save_main_session=True, test_pipeline=None):
  opts = PipelineOptions(argv)
  gem = opts.view_as(GemmaVLLMOptions)
  opts.view_as(SetupOptions).save_main_session = save_main_session

  logging.info(f"Pipeline starting with model path: {gem.model_gcs_path}")

  # vLLM arguments passed directly to the LLM constructor
  handler = VLLMModelHandlerGCS(
      model_gcs_path=gem.model_gcs_path,
      vllm_kwargs={
          "gpu_memory_utilization": 0.7,
          "dtype": "bfloat16", # or "float16"
          "max_num_seqs": 128
      },
  )

  with (test_pipeline or beam.Pipeline(options=opts)) as p:
    processed_elements = (
        p
        # | "ReadPrompts" >> beam.io.ReadFromText(gem.input_file)
        | "ReadPrompts" >> beam.Create(COMPLETION_EXAMPLES)
        | "CountRawReads" >> beam.ParDo(CountFn("pipeline", "prompts_read_from_file"))
        | "NonEmpty" >> beam.Filter(lambda l: l.strip())
        | "CountNonEmpty" >> beam.ParDo(CountFn("pipeline", "prompts_non_empty"))
        | "AddRandomKey" >> beam.Map(lambda x: (random.randint(0, 24), x))
        | "Infer" >> RunInference(handler)
        | "Post" >> beam.ParDo(GemmaPostProcessor())
    )

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