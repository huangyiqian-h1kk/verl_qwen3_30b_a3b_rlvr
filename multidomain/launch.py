"""Subprocess boundary for each model evaluation/training job."""
import argparse


def main():
    from omegaconf import OmegaConf
    from multidomain.train import run
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args()
    run(OmegaConf.load(a.config), a.stage, a.run_dir, a.data_dir)

if __name__ == '__main__':
    main()
