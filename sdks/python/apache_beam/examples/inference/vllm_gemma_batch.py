"""
Batch pipeline: Gemma-2B-it via vLLM → BigQuery.

• Launch from ANY machine (CPU or GPU).  
• Dataflow workers get GPUs via --worker_accelerator.  
• vLLM is imported only on the worker, never on the client.
"""

from __future__ import annotations

import argparse
import logging
import importlib
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.ml.inference.base import (
    ModelHandler,
    PredictionResult,
    RunInference,
)
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)


class GemmaPostProcessor(beam.DoFn):
    def process(
        self, element: tuple[str, PredictionResult]
    ) -> Iterable[dict]:
        prompt, pred = element
        choice = pred.inference.choices[0]
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": choice.logprobs.tokens_consumed.prompt,
            "completion_tokens": choice.logprobs.tokens_consumed.completion,
        }


class LazyVLLMCompletionsHandler(ModelHandler):
    """Wraps Beam's VLLMCompletionsModelHandler but imports vLLM only
    inside `load_model`, i.e. on the Dataflow worker."""
    def __init__(self, model_name: str, vllm_server_kwargs: dict | None = None):
        self._model_name = model_name
        self._vllm_kwargs = vllm_server_kwargs or {}
        self._delegate = None            # will hold actual handler
        self._model = None               # cached engine handle

    # ModelHandler interface
    def load_model(self):
        if self._delegate is None:
            vllm_mod = importlib.import_module(
                "apache_beam.ml.inference.vllm_inference"
            )
            VLLMHandler = getattr(
                vllm_mod, "VLLMCompletionsModelHandler"
            )
            self._delegate = VLLMHandler(
                model_name=self._model_name,
                vllm_server_kwargs=self._vllm_kwargs,
            )
        if self._model is None:
            self._model = self._delegate.load_model()
        return self._model

    def run_inference(self, batch, model, inference_args=None):
        # Re-use the delegate's implementation
        return self._delegate.run_inference(batch, model, inference_args)

    # simply forward to the delegate
    def get_num_bytes(self, batch) -> int:
        return self._delegate.get_num_bytes(batch)

    def batch_elements_kwargs(self):
        return self._delegate.batch_elements_kwargs()


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
        "--model_name", default="google/gemma-2b-it",
        help="HF model checkpoint",
    )
    # Extra perf-test flags (accepted but unused)
    p.add_argument("--publish_to_big_query", default=False)
    p.add_argument("--metrics_dataset")
    p.add_argument("--metrics_table")
    p.add_argument("--influx_measurement")
    p.add_argument("--device", default="GPU")
    return p.parse_known_args(argv)


def run(argv=None, save_main_session=True, test_pipeline=None):
    opts, pipeline_args = _parse(argv)

    handler = LazyVLLMCompletionsHandler(
        model_name=opts.model_name,
        vllm_server_kwargs={"gpu_memory_utilization": "0.9"},
    )

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = False
    pipeline_options.view_as(
        SetupOptions
    ).save_main_session = save_main_session

    with (test_pipeline or beam.Pipeline(options=pipeline_options)) as p:
        (
            p
            | "ReadPrompts" >> beam.io.ReadFromText(opts.input)
            | "StripBlank"  >> beam.Filter(lambda ln: ln.strip())
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
