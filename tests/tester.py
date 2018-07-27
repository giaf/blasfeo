#! /usr/bin/python

import subprocess
import sys
import json

SILENT=1
DONE=0
TOTAL=

#TOTAL=$(${#D_BLAS3_ROUTINES[@]} * ${#TARGETS[@]} + ${#S_BLAS3_ROUTINES[@]} * ${#TARGETS[@]})

def make(cmd="", flags, make_flags):
    default_flags=""

    if SILENT: default_flags += " -s"

    flags = " ".join([f"{k}={v}" for k, v in flags.values()])

    run_cmd = f"make {default_flags} {make_flags} {flags} {cmd}"
    make_process = subprocess.Popen(run_cmd,
        shell=True, stdout=subprocess.PIPE, stderr=sys.stdout.fileno())

    while True:
        line = make_process.stdout.readline()
        if not line: break
        print(line.decode("utf-8").strip())
        sys.stdout.flush()
    #  return fail
    return 1

def test_routines(routine_name, flags){
    # build blasfeo
    # MIN_KERNEL_SIZE=$MIN_KERNEL_SIZE SREF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET
    for routine in routines:


        status = make(make_cmd, flags)
        # update_dblas
        # REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET ROUTINE=$ROUTINE ROUTINE_CLASS=$ROUTINE_CLASS TEST_VERBOSITY=$VERBOSE update_dblas
        DONE+=1

        if not status:
            print(f"Error with {flags['target']}:{routine['name']} ({DONE}/{TOTAL})")

        print(f"Completed {flags['target']}:{routine['name']} ({DONE}/{TOTAL})")

def build_run_recipe(test_run, schema):
    scheduled_routines = set(test_run['routines'])

    recipe = {"routines":{}, flags:test_run["flags"]}

    available_classes = schema['routines']

    for routine_class in available_classes:
        available_subclasses = available_classes[routine_class]

        for routine_subclass in available_subclasses:
            available_routines = available_subclasses[routine_subclass]

            for available_routine in available_routines:

                if available_routine in scheduled_routines:
                    scheduled_routines = scheduled_routines - {available_routine}
                    flags = dict(test_run['flags'])

                    for precision in test_run["precisions"]:

                        routine_name = f"{precision}{available_routine}"
                        routine_fullname = f"blasfeo_{routine_name}"
                        flags["ROUTINE"] = routine_fullname
                        flags["ROUTINE_SUBCLASS"] = routine_subclass
                        make_cmd =

                        recipe["routines"][routine_name] = {
                            "class": routine_class,
                            "subclass":routine_class,
                            "make_cmd": f"update_{precision}{routine_class}",
                            "flags": flags
                        }

    if scheduled_routines:
        print(f"Some routines not found in the schema {scheduled_routines}")


    return recipe


if "__name__"==__main__:

    #  run_target("make TEST_VERBOSITY=2 aux")
    #  run_target("make run_aux")

    # load test run configurations
    with open("tests_schema.json") as f:
        schema = json.load(f)

    with open("batch_tests.json") as f:
        test_run = json.load(f)

    # generate recipe
    recipe = build_run_recipe(test_run, schema)

    # compile main library

    if test_run["global_flags"]["build_libs"]:
        make(flags, make_flags="-C ..")

    if test_run["global_flags"]["build_libs"] and test_run["deploy_libs"]
        make(flags, make_flags="-C ..")

    for la in test_run["las"]:
        print(f"Testing {la}")

        if la=="REFERENCE":
            test_routines(test_run)
            break

        if la=="BLAS_WRAPPER":
            test_routines(test_run)
            break

        for target in test_run["target"]:
            print(f"Testing {target}")
            test_routines(test_run)
