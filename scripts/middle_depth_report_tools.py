"""Read a nominated report folder through native commands and isolated CPython."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile
from time import monotonic, sleep

import psutil

from loopqwen.calculator_tool import tool as calculator
from loopqwen.tool_session import Tool


def bounded_command(args, folder):
    with (
        TemporaryFile() as stdout,
        TemporaryFile() as stderr,
        subprocess.Popen(
            args,
            cwd=folder,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        ) as child,
    ):
        until = monotonic() + 3
        proc = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                memory = proc.memory_info().rss
            except psutil.NoSuchProcess:
                break
            if monotonic() >= until or memory > 128 * 2**20:
                child.kill()
                child.wait()
                return {"error": "Local command exceeded its time or memory allowance"}
            sleep(0.02)
        child.wait()
        stdout.seek(0)
        stderr.seek(0)
        return {
            "exit_code": child.returncode,
            "stdout": stdout.read(8192).decode(errors="replace"),
            "stderr": stderr.read(2048).decode(errors="replace"),
        }


def reports(folder):
    folder = Path(folder).resolve(strict=True)
    files = sorted(
        f.name for f in folder.iterdir() if f.is_file() and not f.is_symlink()
    )
    if set(files) != set(p.name for p in folder.iterdir()):
        raise ValueError("Report directory must contain only plain files")
    binary = Path(sys.executable).resolve()
    runtime = Path(sys.base_prefix).resolve()
    trusted = (
        f'(subpath "/System")(subpath "/usr/lib")(subpath {json.dumps(str(runtime))})'
    )
    profile = (
        '(version 1)(deny default)(import "dyld-support.sb")(allow syscall*)'
        f"(allow process-exec (literal {json.dumps(str(binary))}))(allow sysctl-read)"
        f"(allow file-map-executable {trusted})(allow file-read-metadata)"
        f'(allow file-read* {trusted} (literal "/dev/urandom")(subpath {json.dumps(str(folder))}))'
    )

    def terminal(command, files=None):
        paths = [] if files is None else files
        commands = {"ls": "/bin/ls", "cat": "/bin/cat", "wc": "/usr/bin/wc"}
        if command not in commands or any(f not in names for f in paths):
            raise ValueError("Use a supplied filename and a supported command")
        if command != "ls" and not paths:
            raise ValueError("Select at least one report")
        return bounded_command([commands[command], *paths], folder)

    def python(code):
        # Apply hard limits inside the interpreter before it executes the program.
        limits = (
            "import resource,sys\n"
            "resource.setrlimit(resource.RLIMIT_CPU,(2,2))\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))\n"
            "resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
            "exec(compile(sys.argv[1],'<report>','exec'),{'__name__':'__main__'})"
        )
        return bounded_command(
            [
                "/usr/bin/sandbox-exec",
                "-p",
                profile,
                str(binary),
                "-IBS",
                "-c",
                limits,
                code,
            ],
            folder,
        )

    names = files

    def schema(props, required):
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }

    return [
        Tool(
            "terminal",
            "Run ls, cat or wc on files in the current report folder.",
            schema(
                {
                    "command": {"type": "string", "enum": ["ls", "cat", "wc"]},
                    "files": {
                        "type": "array",
                        "items": {"type": "string", "enum": names},
                    },
                },
                ["command"],
            ),
            terminal,
        ),
        Tool(
            "python",
            "Run Python 3 standard-library code in the report folder. Read local files and print results. The callback returns stdout, stderr and the exit code.",
            schema(
                {"code": {"type": "string", "minLength": 1, "maxLength": 4000}},
                ["code"],
            ),
            python,
        ),
        calculator(),
    ]
