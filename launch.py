"""Usage: xmanager launch launch.py"""
# FIXME: _JOB_NAME="kubric_$(date +"%b%d_%H%M%S" | tr A-Z a-z)"
# FIXME: how to specify which cloud project we are using?

from typing import Sequence

from absl import app
from absl import flags

from google.cloud import aiplatform_v1beta1 as aip

from xmanager import vizier
from xmanager import xm
from xmanager import xm_local

# --- Flags
_RUN_MODE = flags.DEFINE_enum("run_mode", "local", ["local", "remote", "hyper"],
  "Should I run locally, one single copy remotely, or launch a sweep?")
_CONTAINER = flags.DEFINE_string("container_tag",
  "kubricdockerhub/kubruntudev:latest", "base containter")
_REGION = flags.DEFINE_string("region", "us-central1",
  "GCP execution region; WARNING: match region of bucket!")
_JOB_NAME = flags.DEFINE_string("name", "", "Your name.")
_NR_VIDEOS = flags.DEFINE_integer("nr_videos", 0, "Number of videos.", lower_bound=0)
_NR_WORKERS = flags.DEFINE_integer("nr_workers", 0, "Number of workers.", lower_bound=0)
_PREFIX = flags.DEFINE_string("prefix", "gs://research-brain-kubric-xgcp/jobs",
  "where the output of the execution will be saved.")
_PROJECT_D = flags.DEFINE_string("gcp_project", "kubric-xgcp",
  "name of the GCP project that will execute the workload")


# --- Container configuration.
_DOCKERFILE = """
cat > Dockerfile <<EOF
FROM ${SOURCE_TAG}
COPY ${worker_file} /worker/worker.py
WORKDIR /kubric
ENTRYPOINT ["python3", "/worker/worker.py"]
"""


def get_study_spec() -> aip.StudySpec:
  return aip.StudySpec(
      # FIXME: How to port these from the bash version?
      # NOTE: we are executing a parfor [1..N] here, that"s all we need
      # maxTrials: $NR_VIDEOS
      # maxParallelTrials: $NR_WORKERS
      # maxFailedTrials: 1000
      # enableTrialEarlyStopping: False

      parameters=[
          aip.StudySpec.ParameterSpec(
              parameter_id="seed",
              double_value_spec=aip.StudySpec.ParameterSpec.IntegerValueSpec(
                  min_value=-1, max_value=1000000)),
      ],
      metrics=[
          aip.StudySpec.MetricSpec(
              metric_id="answer", goal=aip.StudySpec.MetricSpec.GoalType.MAXIMIZE)
      ])


def main(_: Sequence[str]) -> None:
  with xm_local.create_experiment(experiment_title="kubric_test") as experiment:
    executable_spec = xm.Container(image_path=_CONTAINER.value)

    if _RUN_MODE.value in ["local"]:
      executor = xm_local.Local()  # FIXME: Mount /kubric, ../assets
    else:
      executor = xm_local.Caip(requirements = xm.JobRequirements(location=_REGION.value))

    [executable] = experiment.package([
      xm.Packageable(
          executable_spec=executable_spec,
          executor_spec=executor.Spec()),
    ])

    job = xm.Job(
      executable=executable,
      executor=executor,
      # FIXME: Forward extra args (unparsed vararg) to jobs
      args=[f"--job-dir '{_PREFIX.value}/{_JOB_NAME.value}'"]
    )

    if _RUN_MODE.value == "hyper":
      vizier.VizierExploration(
          experiment=experiment,
          job=job,
          study_factory=vizier.NewStudy(study_spec=get_study_spec()),
          num_trials_total=_NR_VIDEOS.value,
          num_parallel_trial_runs=_NR_WORKERS.value).launch()
    else:
      experiment.add(job)

if __name__ == "__main__":
  app.run(main)
