#!/usr/bin/env python3
"""krunic.py — Launch tunic.py hyperparameter tuning jobs on SkyPilot clusters."""

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="krunic — launch tunic.py on a SkyPilot cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cluster",       type=str, default="skyb",                       dest="cluster",       help="SkyPilot cluster name")
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
    p.add_argument("--s3-path",       type=str, default="PC",                         dest="s3_path",       help="Dataset path inside S3 bucket")
    p.add_argument("--prefix",        type=str, default="tunic",                      dest="prefix",        help="Prefix for output files and S3 paths")
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
        "PREFIX":      args.prefix,
        "RAY_RESULTS": resume_s3,
    }

    setup = (
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "source $HOME/.local/bin/env\n"
        "uv venv --clear ~/venv\n"
        "uv pip install --python ~/venv/bin/python -r ~/sky_workdir/requirements.txt awscli\n"
        "mkdir -p $DATA_DIR && ~/venv/bin/aws s3 sync s3://$BUCKET/$DATASET_S3 $DATA_DIR\n"
        "sudo snap install nvtop\n"
    )

    run = (
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
        "    --device      auto\n"
        "\n"
        "  ~/venv/bin/aws s3 cp $OUTPUT_DIR/${PREFIX}_results.json $RAY_RESULTS/${PREFIX}_results.json\n"
        "else\n"
        "  ~/venv/bin/ray start --address=$HEAD_IP:$RAY_PORT --block\n"
        "fi\n"
    )

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
    sky.launch(
        task,
        cluster_name=args.cluster,
        idle_minutes_to_autostop=None if args.no_autostop else args.idle_minutes,
        retry_until_up=args.spot,
    )


def main():
    args = parse_args()
    data = build_yaml(args)
    yaml_path = save_yaml(args, data)
    launch(args, yaml_path)


if __name__ == "__main__":
    main()
