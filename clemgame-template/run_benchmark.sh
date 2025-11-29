#!/bin/bash
# Usage: scripts/run_benchmark.sh

# do this if running HuggingFace models so that the model weights are downloaded under this directory. For API-based models, it can be ignored.
export HUGGINGFACE_HUB_CACHE=/data/hakimov/huggingface_cache

# set the GPU#0 to be used for inferencing. For API-based models, it can be ignored
export CUDA_VISIBLE_DEVICES=0

# activate  the virtual environment, assuming that the virtual environment files are located under "venv" folder (adjust if it is set up differently)
source venv/bin/activate
export PYTHONPATH=.:$PYTHONPATH

mkdir -p logs


games=(
"taboo"
)

# choose models to run, from global list (clem list models) or local list (the ones in model_registry.json)

models=(
"gpt-4o-2024-08-06"
)

echo
echo "==================================================="
echo "RUNNING: Benchmark Run"
echo "==================================================="
echo


# Runs each model and game separately
for game in "${games[@]}"; do
  for model in "${models[@]}"; do
    echo "Testing ${model} on ${game}"
    # currently, instances.json is the default file for the current run
    # to input a specific instances file, so we could also use the version number here like this: -i instances_file (no need to .json extension)
    { time clem run -g "${game}" -m "${model}" -r "results"; } 2>&1 | tee logs/runtime."${game}"."${model}".log
    # clem run -g "{'benchmark':['2.0']}" -m "${model}"
    { time clem transcribe -g "${game}" -r "results"; } 2>&1 | tee logs/runtime.transcribe."${game}".log
    { time clem score -g "${game}" -r "results"; } 2>&1 | tee logs/runtime.score."${game}".log
  done
done
echo "Evaluating results"
{ time clem eval -r "results"; }


# Runs all games that are part of the benchmark and its specific version, e.g. 2.0 as in example below.
for model in "${models[@]}"; do
  echo "Testing ${model}"
  # currently, instances.json is the default file for the current run
  # to input a specific instances file, so we could also use the version number here like this: -i instances_file (no need to .json extension)
  { time clem run -g "{'benchmark':['2.0']}" -m "${model}" -r "results"; } 2>&1 | tee logs/runtime."${model}".log
  { time clem transcribe -r "results"; } 2>&1 | tee logs/runtime.transcribe.log
  { time clem score -r "results"; } 2>&1 | tee logs/runtime.score.log
done

echo "Evaluating results"
{ time clem eval -r "results"; }

echo "==================================================="
echo "FINISHED: Benchmark Run Version"
echo "==================================================="