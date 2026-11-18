"""Runs inside the SageMaker training and tuning jobs.
SageMaker copies src/ next to this file (the `dependencies` option of the launchers),
so `src` can be imported like on your computer.
"""
from src.train import main

main()
