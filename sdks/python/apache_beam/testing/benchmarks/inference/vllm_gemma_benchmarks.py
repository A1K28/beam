import logging
from apache_beam.examples.inference import vllm_gemma_batch
from apache_beam.testing.load_tests.dataflow_cost_benchmark import DataflowCostBenchmark


class VllmGemmaBenchmarkTest(DataflowCostBenchmark):
    """Launches the Gemma batch pipeline on Dataflow and pushes cost metrics."""

    def __init__(self):
        self.metrics_namespace = "BeamML_vLLM"
        super().__init__(
            metrics_namespace=self.metrics_namespace,
            pcollection="WriteBQ.out0",
        )

    def test(self):
        # The perf-test framework passes --input_file, but the pipeline expects --input.
        extra_opts = {"input": self.pipeline.get_option("input_file")}

        self.result = vllm_gemma_batch.run(
            self.pipeline.get_full_options_as_args(**extra_opts) + [
                "--sdk_worker_parallelism=0",
                "--worker_machine_type=a2-highgpu-1g",
                "--experiments=use_runner_v2,no_use_multiple_sdk_containers,worker_accelerator=type:nvidia-tesla-a100;count:1;install-nvidia-driver",
            ],
            test_pipeline=self.pipeline,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    VllmGemmaBenchmarkTest().run()
