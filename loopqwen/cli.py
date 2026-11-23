import argparse
from .hardware import doctor
from .io import save
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["doctor"])
    parser.parse_args()
    result = doctor()
    save("outputs/doctor.json", result)
    print(json.dumps(result, indent=2))
