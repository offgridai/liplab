#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    print(f"Compare scaffold: {args.baseline} vs {args.candidate}")
    print("Implement after scorecard.csv schema is finalized in liplab_runner.")

if __name__ == "__main__":
    main()
