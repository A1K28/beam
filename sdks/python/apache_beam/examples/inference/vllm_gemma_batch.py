from __future__ import annotations

import faulthandler
faulthandler.enable()

import os
import logging
import tempfile
import time
import random

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.metrics import Metrics

# Import the base class from the Beam SDK's vLLM utility
from apache_beam.ml.inference.vllm_inference import VLLMCompletionsModelHandler

COMPLETION_EXAMPLES = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

class GemmaVLLMOptions(PipelineOptions):
  """Custom pipeline options."""
  @classmethod
  def _add_argparse_args(cls, parser):
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

class CountFn(beam.DoFn):
    """A simple DoFn that counts elements and passes them through."""
    def __init__(self, namespace, name):
        super().__init__()
        self.counter = Metrics.counter(namespace, name)

    def process(self, element):
        self.counter.inc()
        yield element

# --- NEW CLASS: Inherits from Beam's handler to add GCS support ---
class VLLMCompletionsModelHandlerFromGCS(VLLMCompletionsModelHandler):
    """
    A custom model handler that extends Beam's built-in VLLM handler
    to support loading models from a GCS path.
    """
    def __init__(self, model_gcs_path: str, **kwargs):
        """
        Args:
          model_gcs_path: The GCS path to the directory containing model files.
          **kwargs: Additional arguments for VLLMCompletionsModelHandler.
        """
        # Initialize the parent class. We use a placeholder for model_name
        # because the real path will be determined during load_model.
        super().__init__(model_name="placeholder", **kwargs)
        self._model_gcs_path = model_gcs_path
        self._local_path = None # To store the path of the downloaded model

    def load_model(self):
        """
        Overrides the base method to download the model before starting the server.
        """
        # Only download the model once per handler instance.
        if self._local_path is None:
            local_dir = tempfile.mkdtemp()
            logging.info(f"[HANDLER] Downloading model from {self._model_gcs_path} to {local_dir}...")
            
            # Use Beam's FileSystems to handle GCS paths
            gcs_dir = self._model_gcs_path.rstrip('/')
            pattern = f"{gcs_dir}/*"
            match_result = FileSystems.match([pattern])[0]
            
            if not match_result.metadata_list:
                raise RuntimeError(f"No files found matching pattern {pattern}")

            for metadata in match_result.metadata_list:
                src = metadata.path
                dst = os.path.join(local_dir, os.path.basename(src))
                with FileSystems.open(src, 'rb') as fs_read, open(dst, 'wb') as fs_write:
                    fs_write.write(fs_read.read())
            
            logging.info(f"[HANDLER] GCS download complete.")
            self._local_path = local_dir

        # IMPORTANT: Set the parent's _model_name to our new local path.
        # This is the path the vLLM server subprocess will use.
        self._model_name = self._local_path

        # Now, call the original load_model, which will start the server
        # using the local path we just provided.
        return super().load_model()

# --- UPDATED POST-PROCESSOR ---
# The output from the OpenAI server is different from the direct vLLM engine output.
class OpenAICompletionsPostProcessor(beam.DoFn):
  def __init__(self):
    super().__init__()
    self.received_counter = Metrics.counter('PostProcessor', 'elements_received')
    self.success_counter = Metrics.counter('PostProcessor', 'elements_succeeded')
    self.failure_counter = Metrics.counter('PostProcessor', 'elements_failed_or_empty')

  def process(self, element: PredictionResult):
    self.received_counter.inc()
    prompt, openai_completion = element.example, element.inference

    if not openai_completion or not openai_completion.choices:
      self.failure_counter.inc()
      logging.warning(f"Received empty or invalid inference for prompt: {prompt}")
      return

    choice = openai_completion.choices[0]
    usage = openai_completion.usage
    self.success_counter.inc()
    yield {
        "prompt": prompt,
        "completion": choice.text.strip(),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }

def run(argv=None, save_main_session=True, test_pipeline=None):
  opts = PipelineOptions(argv)
  gem = opts.view_as(GemmaVLLMOptions)
  opts.view_as(SetupOptions).save_main_session = save_main_session

  logging.info(f"Pipeline starting with model from GCS: {gem.model_gcs_path}")

  # Pass arguments for the vLLM server here.
  # These are passed as command-line flags to the subprocess.
  vllm_args = {
      'gpu-memory-utilization': '0.80',
      'dtype': 'float16',
  }
  
  # Pass arguments for the model's generate/create call here.
  inference_args = {
      'max_tokens': 1024,
      'temperature': 0.2,
  }

  handler = VLLMCompletionsModelHandlerFromGCS(
      model_gcs_path=gem.model_gcs_path,
      vllm_server_kwargs=vllm_args
  )

  with (test_pipeline or beam.Pipeline(options=opts)) as p:
    (
        p
        # | "ReadPrompts" >> beam.io.ReadFromText(gem.input_file)
        | "ReadPrompts" >> beam.Create(COMPLETION_EXAMPLES)
        | "CountRawReads" >> beam.ParDo(CountFn("pipeline", "prompts_read_from_file"))
        | "RunInference" >> RunInference(handler, inference_args=inference_args)
        | "PostProcess" >> beam.ParDo(OpenAICompletionsPostProcessor())
        | "CountElementsForBQ" >> beam.ParDo(CountFn("pipeline", "elements_to_bq"))
        | "WriteToBQ" >> beam.io.WriteToBigQuery(
            gem.output_table,
            schema="prompt:STRING,completion:STRING,prompt_tokens:INTEGER,completion_tokens:INTEGER",
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            method=beam.io.WriteToBigQuery.Method.FILE_LOADS,
        )
    )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()