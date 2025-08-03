#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Batch pipeline: RunInference ➜ vLLM ➜ Gemma-2B-it ➜ BigQuery.

Reads a text file of prompts from GCS, generates completions with vLLM, and
writes the rows {prompt, completion, prompt_tokens, completion_tokens} to BQ.

The vLLM import happens lazily inside `run()` so that unit tests executed on a
CPU-only host don’t require CUDA libraries.
"""

import argparse
import logging
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.ml.inference.base import PredictionResult, RunInference
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)

import torch


class GemmaPostProcessor(beam.DoFn):
    """Turn vLLM PredictionResult into a BigQuery-ready dict."""

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


def _parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--input_file",
        dest="input",
        required=True,
        help="gs://BUCKET/prompts.txt (one prompt per line)",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="PROJECT:DATASET.gemma_batch",
    )
    parser.add_argument(
        "--model_name", default="google/gemma-2b-it", help="HF model ID"
    )

    # Perf-test harness adds these; accept and ignore.
    parser.add_argument("--publish_to_big_query", default=False)
    parser.add_argument("--metrics_dataset")
    parser.add_argument("--metrics_table")
    parser.add_argument("--influx_measurement")
    parser.add_argument("--device", default="GPU")

    return parser.parse_known_args(argv)


def run(argv=None, save_main_session=True, test_pipeline=None):
    if not torch.cuda.is_available():
        from unittest import SkipTest
        raise SkipTest("CUDA not available -- vLLM requires a GPU runtime")

    opts, pipeline_args = _parse(argv)

    # Lazy import after the CUDA check
    from apache_beam.ml.inference.vllm_inference import \
        VLLMCompletionsModelHandler

    handler = VLLMCompletionsModelHandler(
        model_name=opts.model_name,
        vllm_server_kwargs={"gpu_memory_utilization": "0.9"}
    )

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = False
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with (test_pipeline or beam.Pipeline(options=pipeline_options)) as p:
        (
            p
            | "ReadText"    >> beam.io.ReadFromText(opts.input)
            | "StripBlank"  >> beam.Filter(lambda ln: ln.strip())
            | "RunInference" >> RunInference(handler)
            | "PostProcess" >> beam.ParDo(GemmaPostProcessor())
            | "WriteBQ"     >> beam.io.WriteToBigQuery(
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
