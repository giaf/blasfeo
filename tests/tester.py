#! /usr/bin/python

import subprocess as sp
import jinja2 as jn
import sys
import json
import argparse
import re
from pathlib import Path

BLASFEO_DIR=Path(__file__).absolute().parents[1]
RECIPE_SCHEMA="test_schema.json"
RECIPE_JSON="recipe_default.json"
REPORTS_DIR="reports"
TESTCLASSES_PATH="classes"
TPL_PATH="Makefile.tpl"


def parse_arguments():
    parser = argparse.ArgumentParser(description='BLAFEO tests scheduler')

    parser.add_argument(dest='recipe_json', type=str, default=RECIPE_JSON, nargs='?',
        help='Run a batch of test from a specific recipe, i.e. recipe_all.json')
    parser.add_argument('--silent', default=False, action='store_true',
        help='Silent makefile output')
    parser.add_argument('--run_all', default=False, action='store_true',
        help='Do not interrupt tests sweep on error')
    parser.add_argument('--rebuild', default=False, action='store_true',
        help='Rebuild libblasfeo to take into account recent code '+
        'changes or addition of new target to the recipe batch')

    args = parser.parse_args()
    return args


def make(cmd="", cflags={}, make_flags={}, env_flags={}):

    make_flags = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in make_flags.items()])
    env_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])

    run_cmd = "{env_flags} make {make_flags} {cmd}".format(env_flags=env_flags, make_flags=make_flags, cmd=cmd)

    print(run_cmd,"\n")
    make_process = sp.Popen(run_cmd,
        shell=True, stdout=sp.PIPE, stderr=sys.stdout.fileno())

    while True:
        line = make_process.stdout.readline()
        if not line: break
        print(line.decode("utf-8").strip())
        sys.stdout.flush()
    return make_process.returncode


def make_templated(make_cmd="", cflags={}, env_flags={}, make_flags={}, **kargs):

    run_id = "{la}_{target}_{routine_fullname}"\
        .format(la=make_flags['LA'], target=make_flags['TARGET'], routine_fullname=kargs["fullname"])

    print("\nTesting {run_id}\n".format(run_id=run_id))


    with open(TPL_PATH) as f:
        template = jn.Template(f.read())

    makefile = template.render(cflags=cflags)

    # flag to override default libblasfeo flags
    make_flags_cmd = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in make_flags.items()])

    #  env_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])
    #  run_cmd = "{env_flags} make -f - {cmd}".format(env_flags=env_flags, cmd=cmd)
    run_cmd = "make {make_flags_cmd} -f - {make_cmd}".format(make_cmd=make_cmd, make_flags_cmd=make_flags_cmd)
    report_cmd = "make {make_flags_cmd} {make_cmd}".format(make_cmd=make_cmd, make_flags_cmd=make_flags_cmd)
    make = sp.Popen(run_cmd.split(),
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE
        )

    outs, errs = make.communicate(makefile.encode("utf8"))

    if outs: print("Make infos:\n", outs.decode("utf8"))
    if errs: print("Make errors:\n",errs.decode("utf8"))

    if make.returncode:
        # write report
        report_dir_path = Path(REPORTS_DIR, run_id)
        report_dir_path.mkdir(parents=True, exist_ok=True)
        with open(Path(report_dir_path, "Makefile"), "w") as f:
            f.write(makefile)
        with open(Path(report_dir_path, "make.sh"), "w") as f:
            f.write("#! /bin/bash\n")
            f.write(report_cmd+"\n")

        print("Error with {run_id}".format(run_id=run_id))

    else:
        print("Tested with {run_id}".format(run_id=run_id))

    return make.returncode

class CookBook:
    def __init__(self, cli_flags):

        self.cli_flags=cli_flags

        with open(cli_flags.recipe_json) as f:
            self.specs = json.load(f)

        with open(RECIPE_SCHEMA) as f:
            self.schema = json.load(f)

        self.DONE = 0
        self.ERRORS = 0

        self.TOTAL =\
            len(set(self.specs["routines"]))\
            * len(self.specs["targets"])\
            * len(self.specs["precisions"])\
            * len(self.specs["apis"])

        # build standard recipe skelethon
        self.build_recipe()

    def parse_routine_options(self, routine_name, available_flags):
        pattern = '(?P<routine_basename>[a-z]*)_'
        for flag_name, flags_values in available_flags.items():
            flags_values = '|'.join(flags_values)
            pattern += '(?P<{flag_name}>[{flags_values}])'.format(flag_name=flag_name, flags_values=flags_values)
        parsed_flags = re.search(pattern, routine_name).groupdict()
        return parsed_flags

    def build_recipe(self):
        scheduled_routines = set(self.specs['routines'])

        # create recipe with no global flags
        self.recipe = dict(self.specs)
        self.recipe["scheduled_routines"] = {}
        self.recipe["make_flags"].update({"BLASFEO_DIR":BLASFEO_DIR})

        available_groups = self.schema['routines']

        # routine groups: blas1 blas2 ..
        for group_name, available_classes in available_groups.items():

            # routine classes: gemm, trsm, ...
            for class_name, routine_class in available_classes.items():
                available_routines = routine_class["routines"]
                routine_flags = routine_class["flags"]

                # routines: gemm_nn, gemm_nt, ...
                for routine in available_routines:

                    if routine not in scheduled_routines:
                        continue

                    scheduled_routines = scheduled_routines - {routine}

                    # precision
                    for precision in self.specs["precisions"]:

                        # apis
                        for api in self.specs["apis"]:

                            cflags = {}
                            # build on top of global (for every routine) cflag collection
                            cflags.update(self.specs['cflags'])

                            cflags["ROUTINE_CLASS"] = class_name
                            cflags["PRECISION_{}".format(precision.upper())] = None

                            routine_fullname = "{api}_{precision}{routine}".format(api=api, precision=precision[0], routine=routine)

                            if api=="blas":
                                cflags["TEST_BLAS_API"] = None
                                routine_dict = self.parse_routine_options(routine, routine_flags)
                                cflags.update(routine_dict)
                                routine_testclass_src = "blasapi_"+routine_class["testclass_src"]
                                routine_name = "{precision}{routine}".format(precision=precision[0], routine=class_name)
                            else:
                                routine_testclass_src = routine_class["testclass_src"]
                                routine_name = "{precision}{routine}".format(precision=precision[0], routine=routine)

                            cflags["ROUTINE_CLASS_C"] = Path(TESTCLASSES_PATH, routine_testclass_src)
                            cflags["ROUTINE"] = routine_name
                            cflags["ROUTINE_FULLNAME"] = routine

                            # add blas_api flag arguments values

                            self.recipe["scheduled_routines"][routine_fullname] = {
                                "group": group_name,
                                "class": class_name,
                                "api": api,
                                "precision": precision,
                                "make_cmd": "update",
                                "fullname": routine_fullname,
                                "cflags": cflags
                            }

        if scheduled_routines:
            print("Some routines were not found in the schema ({}) {}"
                  .format(RECIPE_SCHEMA, scheduled_routines))

    def run_all(self):
        # tune the recipe and run

        for la in self.specs["las"]:
            print("Testing {la}".format(la=la))
            self.recipe["make_flags"]["LA"]=la

            if la=="REFERENCE":
                self.run_recipe()
                break

            if la=="BLAS_WRAPPER":
                self.run_recipe()
                break

            for target in self.specs["targets"]:
                print("Testing {target}".format(target=target))

                self.recipe["make_flags"]["TARGET"]=target
                self.run_recipe()

    def is_compiled(self):

        target = self.recipe["make_flags"]["TARGET"]
        la = self.recipe["make_flags"]["LA"]

        _path_lib_static = Path("build", la, target, "libblasfeo.a")
        _path_libref_static = Path("build", la, target, "libblasfeo_ref.a")

        if _path_lib_static.is_file() and _path_libref_static.is_file():
            return 1

        return 0

    def run_recipe(self):
        # preparation step
        cflags = self.recipe["cflags"]
        make_flags = self.recipe["make_flags"]
        env_flags = self.recipe["env_flags"]

        if self.cli_flags.silent or self.recipe["env_flags"].get("silent"):
            make_flags.update({"-s":None})

        if self.cli_flags.rebuild or not self.is_compiled():
            libblasfeo_flags = dict(make_flags)
            libblasfeo_flags.update({"BLAS_API":1})
            make("-C .. ", make_flags=libblasfeo_flags)
            make("-C .. deploy_to_tests", make_flags=libblasfeo_flags)

        for routine_name, args in self.recipe['scheduled_routines'].items():
            # update local flags with global flags

            if args.get("cflags"):
                args["cflags"].update(cflags)
            else:
                args["flags"] = cflags

            if args.get("env_flags"):
                args["env_flags"].update(env_flags)
            else:
                args["env_flags"] = env_flags

            if args.get("make_flags"):
                args["make_flags"].update(make_flags)
            else:
                args["make_flags"] = make_flags

            error =  self.test_routine(routine_name, args)

            if error and not self.cli_flags.run_all: break

    def test_routine(self, routine_fullname, kargs):

        error = make_templated(**kargs)

        if not error:
            self.DONE += 1
        else:
            self.ERRORS += 1

        print("({done}:Succeded, {errors}:Errors) / ({total}:Total)"
            .format(done=self.DONE, errors=self.ERRORS, total=self.TOTAL))

        return error



if __name__ == "__main__":

    cli_flags = parse_arguments()

    # generate recipes
    # test set to be run in the given excution of the script
    cookbook = CookBook(cli_flags)
    #  print(json.dumps(cookbook.recipe, indent=4))
    cookbook.run_all()
