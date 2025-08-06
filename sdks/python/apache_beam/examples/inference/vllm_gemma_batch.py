from __future__ import annotations
  
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import WorkerOptions

import logging
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.vllm_inference import VLLMCompletionsModelHandler
from apache_beam.ml.inference.base import PredictionResult
import apache_beam as beam

class FormatOutput(beam.DoFn):
    def process(self, element, *args, **kwargs):
        yield "Input: {input}, Output: {output}".format(input=element.example, output=element.inference)

def run(argv=None, save_main_session=True, test_pipeline=None):
    options = PipelineOptions()

    # Replace with the image repository and tag from the previous step.
    CONTAINER_IMAGE = 'us.gcr.io/apache-beam-testing/beam-vllm-gpu-base:v0.3.0_NVIDIA'  # @param {type:'string'}
    # Replace with your GCP project
    PROJECT_NAME = 'apache-beam-testing' # @param {type:'string'}

    options.view_as(GoogleCloudOptions).project = PROJECT_NAME

    # Provide required pipeline options for the Dataflow Runner.
    options.view_as(StandardOptions).runner = "DataflowRunner"

    # Set the Google Cloud region that you want to run Dataflow in.
    options.view_as(GoogleCloudOptions).region = 'us-central1'

    # IMPORTANT: Replace BUCKET_NAME with the name of your Cloud Storage bucket.
    dataflow_gcs_location = "gs://temp-storage-for-perf-tests/loadtests"

    # The Dataflow staging location. This location is used to stage the Dataflow pipeline and the SDK binary.
    options.view_as(GoogleCloudOptions).staging_location = '%s/staging' % dataflow_gcs_location

    # The Dataflow staging location. This location is used to stage the Dataflow pipeline and the SDK binary.
    options.view_as(GoogleCloudOptions).staging_location = '%s/staging' % dataflow_gcs_location

    # The Dataflow temp location. This location is used to store temporary files or intermediate results before outputting to the sink.
    options.view_as(GoogleCloudOptions).temp_location = '%s/temp' % dataflow_gcs_location

    # Enable GPU runtime. Make sure to enable 5xx driver since vLLM only works with 5xx drivers, not 4xx
    options.view_as(GoogleCloudOptions).dataflow_service_options = ["worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver:5xx"]

    options.view_as(SetupOptions).save_main_session = True

    # Choose a machine type compatible with GPU type
    options.view_as(WorkerOptions).machine_type = "n1-standard-4"

    options.view_as(WorkerOptions).sdk_container_image = CONTAINER_IMAGE

    logging.getLogger().setLevel(logging.INFO)  # Output additional Dataflow Job metadata and launch logs. 
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "John cena is",
    ]

    # Specify the model handler, providing a path and the custom inference function.
    model_handler = VLLMCompletionsModelHandler('facebook/opt-125m')

    with beam.Pipeline(options=options) as p:
        _ = (p | beam.Create(prompts) # Create a PCollection of the prompts.
                | RunInference(model_handler) # Send the prompts to the model and get responses.
                | beam.ParDo(FormatOutput()) # Format the output.
                | beam.Map(logging.info) # Print the formatted output.
        )
        
    return p.result


# def run(argv=None, save_main_session=True, test_pipeline=None):
#   # Build pipeline options
#   opts = PipelineOptions(argv)

#   gem = opts.view_as(GemmaVLLMOptions)
#   opts.view_as(SetupOptions).save_main_session = False

#   logging.info(f"Pipeline starting with model path: {gem.model_gcs_path}")
#   handler = VLLMModelHandlerGCS(
#       model_gcs_path=gem.model_gcs_path,
#       vllm_kwargs={
#           "gpu_memory_utilization": 0.7,
#           "dtype": "float16",
#           "max_num_seqs": 128
#       },
#   )

#   with (test_pipeline or beam.Pipeline(options=opts)) as p:
#     processed_elements = (
#         p
#         # For testing, we use the in-memory list. To use the file, uncomment
#         # the line below and comment out the beam.Create line.
#         # | "ReadPrompts" >> beam.io.ReadFromText(gem.input_file)
#         | "ReadPrompts" >> beam.Create(COMPLETION_EXAMPLES)
#         | "CountRawReads" >> beam.ParDo(CountFn("pipeline", "prompts_read"))
#         | "NonEmpty" >> beam.Filter(lambda l: l.strip())
#         | "CountNonEmpty" >> beam.ParDo(CountFn("pipeline", "prompts_non_empty"))
#         # Using 25 keys as an example, matching the max_num_workers.
#         | "AddRandomKey" >> beam.Map(lambda x: (random.randint(0, 24), x))
#         | "Infer" >> RunInference(handler)
#         # PostProcessor now has its own internal counters
#         | "Post" >> beam.ParDo(GemmaPostProcessor())
#     )

#     # Add a final counter before writing to BQ to see what is being loaded
#     (
#         processed_elements
#         | "CountElementsForBQ" >> beam.ParDo(CountFn("pipeline", "elements_to_bq"))
#         | "WriteToBQ" >> beam.io.WriteToBigQuery(
#             gem.output_table,
#             schema=
#             "prompt:STRING,completion:STRING,prompt_tokens:INTEGER,completion_tokens:INTEGER",
#             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
#             create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
#             method=beam.io.WriteToBigQuery.Method.FILE_LOADS,
#         ))

#   return p.result


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    logging.getLogger().setLevel(logging.INFO)
    run()