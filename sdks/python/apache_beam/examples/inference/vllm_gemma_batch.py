"""
Batch pipeline: RunInference + vLLM (OpenAI-compatible) + Gemma-2B-it.

Reads a text file of prompts from GCS, generates completions with vLLM,
and appends rows to BigQuery.

Schema: prompt, completion, prompt_tokens, completion_tokens
"""

import argparse
import logging
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.ml.inference.base import RunInference, PredictionResult
from apache_beam.ml.inference.vllm_inference import VLLMCompletionsModelHandler
from apache_beam.options.pipeline_options import (
    PipelineOptions, SetupOptions, StandardOptions)


class GemmaPostProcessor(beam.DoFn):
    """Convert PredictionResult → row that BigQuery accepts."""
    def process(self,
                element: tuple[str, PredictionResult]) -> Iterable[dict]:
        prompt, pred = element
        choice = pred.inference.choices[0]  # first (and only) completion
        yield {
            "prompt": prompt,
            "completion": choice.text.strip(),
            "prompt_tokens": choice.logprobs.tokens_consumed.prompt,
            "completion_tokens": choice.logprobs.tokens_consumed.completion,
        }


def _parse(argv=None):
    # Accept both --input  *and*  --input_file so the perf-test harness works
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--input_file", dest="input", required=True,
        help="gs://…/sentences_50k.txt (one prompt per line)")

    p.add_argument("--output_table", required=True,
        help="PROJECT:DATASET.gemma_batch")
    p.add_argument("--model_name", default="google/gemma-2b-it")

    # Flags that the load-test framework appends; we just swallow them so
    # PipelineOptions passes them through harmlessly.
    p.add_argument("--publish_to_big_query", default=False)
    p.add_argument("--metrics_dataset")
    p.add_argument("--metrics_table")
    p.add_argument("--influx_measurement")
    p.add_argument("--device", default="GPU")   # GPU is required for vLLM

    return p.parse_known_args(argv)


def run(argv=None, save_main_session=True):
    opts, pipeline_args = _parse(argv)

    # vLLM handler – spins up the model server in each worker
    handler = VLLMCompletionsModelHandler(
        model_name=opts.model_name,
        vllm_server_kwargs={"gpu_memory_utilization": "0.9"})

    # Batch-only flags
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = False
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "Read" >> beam.io.ReadFromText(opts.input)
            | "DropBlank" >> beam.Filter(lambda ln: ln.strip())
            | "RunInference" >> RunInference(handler)
            | "ToBQRow" >> beam.ParDo(GemmaPostProcessor())
            | "WriteBQ" >> beam.io.WriteToBigQuery(
                opts.output_table,
                schema=("prompt:STRING, completion:STRING,"
                        "prompt_tokens:INTEGER, completion_tokens:INTEGER"),
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                method=beam.io.WriteToBigQuery.Method.FILE_LOADS)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
