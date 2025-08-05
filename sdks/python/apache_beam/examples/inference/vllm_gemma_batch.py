#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" A sample pipeline using the RunInference API to interface with an LLM using
vLLM. Takes in a set of prompts or lists of previous messages and produces
responses using a model of choice.

Requires a GPU runtime with vllm, openai, and apache-beam installed to run
correctly, and your local vllm_inference.py on PYTHONPATH.
"""

import argparse
import logging
from collections.abc import Iterable

import apache_beam as beam
from apache_beam.ml.inference.base import PredictionResult, RunInference
from apache_beam.ml.inference.vllm_inference import VLLMChatModelHandler, VLLMCompletionsModelHandler
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.runners.runner import PipelineResult

COMPLETION_EXAMPLES = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "John cena is",
]

def parse_known_args(argv):
    """
    Parses args for the workflow; everything else (runner, region, etc.)
    is passed through to Dataflow via PipelineOptions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=True,
        help='GCS URI of newline-delimited text inputs')
    parser.add_argument(
        '--model_name',
        dest='model_name',
        required=True,
        help='Name of the vLLM model (e.g. google/gemma-2b-it)')
    parser.add_argument(
        '--output',
        dest='output',
        required=False,
        help='Local/text output path (ignored if publishing to BigQuery)')
    parser.add_argument(
        '--chat',
        action='store_true',
        help='If set, use the chat model handler instead of completions')
    parser.add_argument(
        '--chat_template',
        dest='chat_template',
        help='GCS path to your chat template (only if --chat)')
    # all other flags (runner, region, etc.) are consumed by PipelineOptions
    return parser.parse_known_args(argv)


class PostProcessor(beam.DoFn):
    def process(self, element: PredictionResult) -> Iterable[str]:
        yield str(element.example) + ": " + str(element.inference)


def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    known_args, pipeline_args = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # choose model handler
    if known_args.chat:
        model_handler = VLLMChatModelHandler(
            model_name=known_args.model_name,
            chat_template_path=known_args.chat_template)
    else:
        model_handler = VLLMCompletionsModelHandler(
            model_name=known_args.model_name)

    # build pipeline
    pipeline = test_pipeline if test_pipeline else beam.Pipeline(options=pipeline_options)

    # examples = pipeline | "ReadInput" >> beam.io.ReadFromText(known_args.input)
    examples = pipeline | "ReadInput" >> beam.Create(COMPLETION_EXAMPLES)

    predictions = examples | "RunInference" >> RunInference(model_handler)
    process_output = predictions | "Process Predictions" >> beam.ParDo(PostProcessor())

    opts = pipeline_options.get_all_options()
    if opts.get('publish_to_big_query'):
        process_output | "WriteToBQ" >> beam.io.WriteToBigQuery(
            table=opts['metrics_table'],
            dataset=opts['metrics_dataset'],
            schema='example:STRING,inference:STRING',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
    else:
        process_output | "WriteOutput" >> beam.io.WriteToText(
            known_args.output,
            shard_name_template='',
            append_trailing_newlines=True)

    result = pipeline.run()
    result.wait_until_finish()
    return result


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
