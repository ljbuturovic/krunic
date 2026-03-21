#!/usr/bin/env python3
"""krunic.py — Launch tunic.py hyperparameter tuning jobs on SkyPilot clusters."""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import yaml


class _LiteralStr(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_LiteralStr, _literal_representer)


def parse_args():
    p = argparse.ArgumentParser(
        description="krunic — launch tunic.py on a SkyPilot cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cluster",       type=str, required=True,                        dest="cluster",       help="SkyPilot cluster name")
    p.add_argument("--cloud",         type=str, default="aws",                        dest="cloud",         help="Cloud provider")
    p.add_argument("--accelerator",   type=str, default="T4:4",                       dest="accelerator",   help="Accelerator spec (e.g. T4:4, A10G:1)")
    p.add_argument("--num-nodes",     type=int, default=1,                            dest="num_nodes",     help="Number of cluster nodes")
    p.add_argument("--disk-size",     type=int, default=200,                          dest="disk_size",     help="Disk size in GB per node")
    p.add_argument("--spot",          action="store_true",                            dest="spot",          help="Use spot instances")
    p.add_argument("--bucket",        type=str, default="image.data",                 dest="bucket",        help="S3 bucket name")
    p.add_argument("--requirements",  type=str, default="~/github/tunic/requirements.txt", dest="requirements", help="Local requirements.txt path")
    p.add_argument("--model",         type=str, default="resnet50",                   dest="model",         help="timm model name")
    p.add_argument("--n-trials",      type=int, default=30,                           dest="n_trials",      help="Number of Optuna trials")
    p.add_argument("--n-epochs",      type=int, default=30,                           dest="n_epochs",      help="Training epochs per trial")
    p.add_argument("--s3-path",       type=str, required=True,                        dest="s3_path",       help="Dataset path inside S3 bucket")
    p.add_argument("--prefix",        type=str, default="tunic",                      dest="prefix",        help="Prefix for output files and S3 paths")
    p.add_argument("--training-fraction", type=float, default=1.0,                   dest="training_fraction", help="Fraction of training data to use (e.g. 0.1 for 10%%); same subset across all trials")
    p.add_argument("--idle-minutes",  type=int, default=60,                           dest="idle_minutes",  help="Auto-stop cluster after N idle minutes")
    p.add_argument("--no-autostop",   action="store_true",                            dest="no_autostop",   help="Disable auto-stop; cluster stays up after job finishes")
    return p.parse_args()


def build_yaml(args) -> dict:
    requirements_path = Path(args.requirements).expanduser()
    workdir = str(requirements_path.parent)

    resources = {
        "cloud": args.cloud,
        "accelerators": args.accelerator,
        "disk_size": args.disk_size,
    }
    if args.spot:
        resources["use_spot"] = True

    resume_s3 = f"s3://{args.bucket}/ray-results/{args.prefix}"

    envs = {
        "BUCKET":      args.bucket,
        "DATASET_S3":  args.s3_path,
        "DATA_DIR":    "/home/ubuntu/data/dataset",
        "OUTPUT_DIR":  "/home/ubuntu/data/output",
        "MODEL":       args.model,
        "N_TRIALS":    str(args.n_trials),
        "EPOCHS":      str(args.n_epochs),
        "PREFIX":             args.prefix,
        "RAY_RESULTS":        resume_s3,
        "TRAINING_FRACTION":  str(args.training_fraction),
    }

    setup = _LiteralStr(textwrap.dedent("""\
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        uv venv --clear ~/venv
        uv pip install --python ~/venv/bin/python -r ~/sky_workdir/requirements.txt awscli
        mkdir -p $DATA_DIR
        if awk "BEGIN{exit !($TRAINING_FRACTION < 1.0)}"; then
          for split in train val test; do
            CLASSES=$(~/venv/bin/aws s3 ls s3://$BUCKET/$DATASET_S3/$split/ 2>/dev/null | awk '/PRE/{print $2}' | sed 's|/$||')
            [ -z "$CLASSES" ] && continue
            while IFS= read -r cls; do
              FILES=$(~/venv/bin/aws s3 ls s3://$BUCKET/$DATASET_S3/$split/$cls/ 2>/dev/null | awk '!/PRE/{print $4}')
              [ -z "$FILES" ] && continue
              NFILES=$(echo "$FILES" | wc -l)
              NTAKE=$(awk "BEGIN{n=int($NFILES * $TRAINING_FRACTION); print (n<1)?1:n}")
              mkdir -p "$DATA_DIR/$split/$cls"
              echo "  [$split/$cls] copying $NTAKE / $NFILES files..."
              (
                while true; do
                  DONE=$(find "$DATA_DIR/$split/$cls" -type f 2>/dev/null | wc -l)
                  PCT=$(awk -v done="$DONE" -v ntake="$NTAKE" 'BEGIN{printf "%.0f", (done/ntake)*100}')
                  echo "  [$split/$cls] downloaded $DONE / $NTAKE (${PCT}%)"
                  [ "$DONE" -ge "$NTAKE" ] && break
                  sleep 5
                done
              ) &
              PROGRESS_PID=$!
              echo "$FILES" | shuf | head -n "$NTAKE" | \
                xargs -P 8 -I{} ~/venv/bin/aws s3 cp \
                  "s3://$BUCKET/$DATASET_S3/$split/$cls/{}" \
                  "$DATA_DIR/$split/$cls/{}" --quiet
              kill $PROGRESS_PID 2>/dev/null
              wait $PROGRESS_PID 2>/dev/null
              echo "  [$split/$cls] done ($NTAKE files)"
            done <<< "$CLASSES"
          done
        else
          ~/venv/bin/aws s3 sync s3://$BUCKET/$DATASET_S3 $DATA_DIR
        fi
        sudo snap install nvtop
        sudo snap install btop
    """))

    run = _LiteralStr(textwrap.dedent("""\
        RAY_PORT=6385
        HEAD_IP=$(echo "$SKYPILOT_NODE_IPS" | head -1)

        if [ "$SKYPILOT_NODE_RANK" -eq 0 ]; then
          ~/venv/bin/ray start --head --port=$RAY_PORT
          sleep 10

          mkdir -p $OUTPUT_DIR

          ~/venv/bin/python ~/sky_workdir/tunic.py \\
            --data        $DATA_DIR \\
            --model       $MODEL \\
            --n_trials    $N_TRIALS \\
            --epochs      $EPOCHS \\
            --output      $OUTPUT_DIR/${PREFIX}_results.json \\
            --ray-storage $RAY_RESULTS \\
            --ray-address localhost:$RAY_PORT \\
            --device      auto

          ~/venv/bin/aws s3 cp $OUTPUT_DIR/${PREFIX}_results.json $RAY_RESULTS/${PREFIX}_results.json
        else
          ~/venv/bin/ray start --address=$HEAD_IP:$RAY_PORT --block
        fi
    """))

    return {
        "name": args.cluster,
        "num_nodes": args.num_nodes,
        "resources": resources,
        "envs": envs,
        "setup": setup,
        "run": run,
        "workdir": workdir,
    }


def save_yaml(args, data: dict) -> Path:
    yaml_path = Path(f"{args.prefix}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"YAML written to {yaml_path}")
    return yaml_path


def launch(args, yaml_path: Path):
    try:
        import sky
    except ImportError:
        print("ERROR: skypilot is not installed. Install with: pip install skypilot", file=sys.stderr)
        sys.exit(1)

    task = sky.Task.from_yaml(str(yaml_path))
    print(f"Launching cluster '{args.cluster}' ({args.num_nodes}x {args.accelerator} on {args.cloud}{'  [spot]' if args.spot else ''})...")
    request_id = sky.launch(
        task,
        cluster_name=args.cluster,
        idle_minutes_to_autostop=None if args.no_autostop else args.idle_minutes,
        retry_until_up=args.spot,
    )
    sky.stream_and_get(request_id)


def main():
    args = parse_args()
    data = build_yaml(args)
    yaml_path = save_yaml(args, data)
    launch(args, yaml_path)


if __name__ == "__main__":
    main()
