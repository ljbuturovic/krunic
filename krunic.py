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
    p.add_argument("--requirements",  type=str, default=None,                               dest="requirements", help="Local requirements.txt path (default: <workdir>/requirements.txt)")
    p.add_argument("--workdir",       type=str, required=True,                              dest="workdir",       help="Local directory to sync to the cluster")
    p.add_argument("--model",         type=str, default="resnet50",                   dest="model",         help="timm model name")
    p.add_argument("--n-trials",      type=int, default=30,                           dest="n_trials",      help="Number of Optuna trials")
    p.add_argument("--n-epochs",      type=int, default=30,                           dest="n_epochs",      help="Training epochs per trial")
    p.add_argument("--s3-path",       type=str, required=True,                        dest="s3_path",       help="Dataset path inside S3 bucket")
    p.add_argument("--prefix",        type=str, default="tunic",                      dest="prefix",        help="Prefix for output files and S3 paths")
    p.add_argument("--training-fraction", type=float, default=1.0,                   dest="training_fraction", help="Fraction of training data to use (e.g. 0.1 for 10%%); same subset across all trials")
    p.add_argument("--idle-minutes",  type=int, default=60,                           dest="idle_minutes",  help="Auto-stop cluster after N idle minutes")
    p.add_argument("--copy",          action="store_true",                            dest="copy",          help="Copy data from S3 to local disk instead of mounting (slower setup, faster training)")
    p.add_argument("--no-autostop",   action="store_true",                            dest="no_autostop",   help="Disable auto-stop; cluster stays up after job finishes")
    p.add_argument("--tune-metric",   type=str, default="val_auroc",                  dest="tune_metric",   help="Metric used by Optuna and ASHA for trial selection (default: val_auroc)")
    p.add_argument("--batch-size",    type=int, default=32,                            dest="batch_size",    help="Batch size per trial (default: 32)")
    return p.parse_args()


def build_yaml(args) -> dict:
    workdir = str(Path(args.workdir).expanduser())
    requirements_path = Path(args.requirements).expanduser() if args.requirements else Path(workdir) / "requirements.txt"

    resources = {
        "cloud": args.cloud,
        "accelerators": args.accelerator,
        "disk_size": args.disk_size,
    }
    if args.spot:
        resources["use_spot"] = True

    resume_s3 = f"s3://{args.bucket}/ray-results/{args.prefix}"

    _MOUNT_POINT = "/home/ubuntu/s3mount"
    if not args.copy:
        data_dir = f"{_MOUNT_POINT}/{args.s3_path}"
    else:
        data_dir = "/home/ubuntu/data/dataset"

    envs = {
        "BUCKET":             args.bucket,
        "DATASET_S3":         args.s3_path,
        "DATA_DIR":           data_dir,
        "OUTPUT_DIR":         "/home/ubuntu/data/output",
        "MODEL":              args.model,
        "N_TRIALS":           str(args.n_trials),
        "EPOCHS":             str(args.n_epochs),
        "PREFIX":             args.prefix,
        "RAY_RESULTS":        resume_s3,
        "TRAINING_FRACTION":  str(args.training_fraction),
        "TUNE_METRIC":        args.tune_metric,
        "BATCH_SIZE":         str(args.batch_size),
    }

    _setup_start = (
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "source $HOME/.local/bin/env\n"
        "uv venv --clear ~/venv\n"
        "uv pip install --python ~/venv/bin/python -r ~/sky_workdir/requirements.txt awscli\n"
    )
    _setup_s3_copy = (
        "mkdir -p $DATA_DIR\n"
        "if awk \"BEGIN{exit !($TRAINING_FRACTION < 1.0)}\"; then\n"
        "  for split in train val test; do\n"
        "    CLASSES=$(~/venv/bin/aws s3 ls s3://$BUCKET/$DATASET_S3/$split/ 2>/dev/null | awk '/PRE/{print $2}' | sed 's|/$||')\n"
        "    [ -z \"$CLASSES\" ] && continue\n"
        "    while IFS= read -r cls; do\n"
        "      FILES=$(~/venv/bin/aws s3 ls s3://$BUCKET/$DATASET_S3/$split/$cls/ 2>/dev/null | awk '!/PRE/{print $4}')\n"
        "      [ -z \"$FILES\" ] && continue\n"
        "      NFILES=$(echo \"$FILES\" | wc -l)\n"
        "      NTAKE=$(awk \"BEGIN{n=int($NFILES * $TRAINING_FRACTION); print (n<1)?1:n}\")\n"
        "      mkdir -p \"$DATA_DIR/$split/$cls\"\n"
        "      echo \"  [$split/$cls] copying $NTAKE / $NFILES files...\"\n"
        "      (\n"
        "        while true; do\n"
        "          DONE=$(find \"$DATA_DIR/$split/$cls\" -type f 2>/dev/null | wc -l)\n"
        "          PCT=$(awk -v done=\"$DONE\" -v ntake=\"$NTAKE\" 'BEGIN{printf \"%.0f\", (done/ntake)*100}')\n"
        "          echo \"  [$split/$cls] downloaded $DONE / $NTAKE (${PCT}%)\"\n"
        "          [ \"$DONE\" -ge \"$NTAKE\" ] && break\n"
        "          sleep 5\n"
        "        done\n"
        "      ) &\n"
        "      PROGRESS_PID=$!\n"
        "      echo \"$FILES\" | shuf | head -n \"$NTAKE\" | \\\n"
        "        xargs -P 8 -I{} ~/venv/bin/aws s3 cp \\\n"
        "          \"s3://$BUCKET/$DATASET_S3/$split/$cls/{}\" \\\n"
        "          \"$DATA_DIR/$split/$cls/{}\" --quiet\n"
        "      kill $PROGRESS_PID 2>/dev/null\n"
        "      wait $PROGRESS_PID 2>/dev/null\n"
        "      echo \"  [$split/$cls] done ($NTAKE files)\"\n"
        "    done <<< \"$CLASSES\"\n"
        "  done\n"
        "else\n"
        "  ~/venv/bin/aws s3 sync s3://$BUCKET/$DATASET_S3 $DATA_DIR\n"
        "fi\n"
    )
    _setup_end = (
        "sudo snap install nvtop\n"
        "sudo snap install btop\n"
    )

    if not args.copy:
        setup = _LiteralStr(_setup_start + _setup_end)
    else:
        setup = _LiteralStr(_setup_start + _setup_s3_copy + _setup_end)

    _training_fraction_arg = (
        "            --training_fraction $TRAINING_FRACTION \\\n"
        if not args.copy else ""
    )
    run = _LiteralStr(
        "RAY_PORT=6385\n"
        "HEAD_IP=$(echo \"$SKYPILOT_NODE_IPS\" | head -1)\n"
        "\n"
        "if [ \"$SKYPILOT_NODE_RANK\" -eq 0 ]; then\n"
        "  ~/venv/bin/ray start --head --port=$RAY_PORT\n"
        "  sleep 10\n"
        "\n"
        "  mkdir -p $OUTPUT_DIR\n"
        "\n"
        "  ~/venv/bin/python ~/sky_workdir/tunic.py \\\n"
        "    --data        $DATA_DIR \\\n"
        "    --model       $MODEL \\\n"
        "    --n_trials    $N_TRIALS \\\n"
        "    --epochs      $EPOCHS \\\n"
        "    --output      $OUTPUT_DIR/${PREFIX}_results.json \\\n"
        "    --ray-storage $RAY_RESULTS \\\n"
        "    --ray-address localhost:$RAY_PORT \\\n"
        + _training_fraction_arg +
        "    --tune-metric $TUNE_METRIC \\\n"
        "    --batch-size  $BATCH_SIZE \\\n"
        "    --device      auto\n"
        "\n"
        "  ~/venv/bin/aws s3 cp $OUTPUT_DIR/${PREFIX}_results.json $RAY_RESULTS/${PREFIX}_results.json\n"
        "  ~/venv/bin/python -c \"\n"
        "import ray\n"
        "@ray.remote\n"
        "class _Done:\n"
        "    pass\n"
        "ray.init(address='localhost:$RAY_PORT', namespace='tunic', ignore_reinit_error=True)\n"
        "_Done.options(name='tunic_done', lifetime='detached').remote()\n"
        "ray.get_actor('tunic_done')  # block until GCS has registered the actor\n"
        "\"\n"
        "else\n"
        "  ~/venv/bin/ray start --address=$HEAD_IP:$RAY_PORT\n"
        "  ~/venv/bin/python -c \"\n"
        "import ray, time\n"
        "ray.init(address='$HEAD_IP:$RAY_PORT', namespace='tunic', ignore_reinit_error=True)\n"
        "while True:\n"
        "    try:\n"
        "        ray.get_actor('tunic_done')\n"
        "        break\n"
        "    except ValueError:\n"
        "        time.sleep(10)\n"
        "\"\n"
        "fi\n"
    )

    result = {
        "name": args.cluster,
        "num_nodes": args.num_nodes,
        "resources": resources,
        "envs": envs,
        "setup": setup,
        "run": run,
        "workdir": workdir,
    }
    if not args.copy:
        result["file_mounts"] = {
            _MOUNT_POINT: {
                "source": f"s3://{args.bucket}",
                "mode": "MOUNT",
            }
        }
    return result


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
