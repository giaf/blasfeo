#! /usr/bin/python

import subprocess as sp
import jinja2 as jn
import sys
import json
import argparse
import re
from os.path import join

RECIPE_SCHEMA="test_schema.json"
RECIPE_JSON="recipe_default.json"
TESTCLASSES_PATH="classes"
TPL_PATH="Makefile.tpl"

def parse_arguments():
    parser = argparse.ArgumentParser(description='BLAFEO tests scheduler')

    parser.add_argument(dest='recipe_json', type=str, default=RECIPE_JSON, nargs='?',
        help='Run a batch of test from a specific recipe, i.e. recipe_all.json')
    parser.add_argument('--verbose', type=int, default=0,
        help='Verbosity level')
    parser.add_argument('--rebuild', default=False, action='store_true',
        help='Rebuild libblasfeo to take into account recent code '+
        'changes or addition of new target to the recipe batch')

    args = parser.parse_args()
    return args


def make_templated(cmd="", cflags={}, env_flags={}):

    with open(TPL_PATH) as f:
        template = jn.Template(f.read())

    makefile = template.render(cflags=cflags)

    env_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])
    #  run_cmd = "{env_flags} make -f - {cmd}".format(env_flags=env_flags, cmd=cmd)
    run_cmd = "make -s -f - {cmd}".format(cmd=cmd)
    print(run_cmd)

    make = sp.Popen(run_cmd.split(),
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE
        )

    outs, errs = make.communicate(makefile.encode("utf8"))
    print(outs.decode("utf8"))
    print(errs.decode("utf8"))

    return 1


def make(cmd="", cflags={}, env_flags={}):

    cflags = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in cflags.items()])
    env_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])

    run_cmd = "{env_flags} make {cflags} {cmd}".format(env_flags=env_flags, cflags=cflags, cmd=cmd)

    print(run_cmd,"\n")
    make_process = sp.Popen(run_cmd,
        shell=True, stdout=sp.PIPE, stderr=sys.stdout.fileno())

    while True:
        line = make_process.stdout.readline()
        if not line: break
        print(line.decode("utf-8").strip())
        sys.stdout.flush()
    return 1


class CookBook:
    def __init__(self, cli_flags):

        self.cli_flags=cli_flags

        with open(cli_flags.recipe_json) as f:
            self.specs = json.load(f)

        with open(RECIPE_SCHEMA) as f:
            self.schema = json.load(f)

        self.DONE = 0

        self.VERBOSE = self.cli_flags.verbose

        self.TOTAL =\
            len(set(self.specs["routines"]))\
            * len(self.specs["targets"])\
            * len(self.specs["precisions"])\
            * len(self.specs["apis"])

        # build standard recipe skelethon
        self.build_recipe()


    def parse_flags(self, routine_name, available_flags):
        pattern = '(?P<routine_basename>[a-z]*)_'
        for flag_name, flags_values in available_flags.items():
            flags_values = '|'.join(flags_values)
            pattern += '(?P<{flag_name}>[{flags_values}])'.format(flag_name=flag_name, flags_values=flags_values)
        parsed_flags = re.search(pattern, routine_name).groupdict()
        return parsed_flags

    def build_recipe(self):
        scheduled_routines = set(self.specs['routines'])

        # create recipe with no global flags
        self.recipe = {
            'routines':{},
            'cflags':self.specs['cflags'],
            'env_flags':self.specs['env_flags']
        }


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

                            flags = {}
                            flags.update(self.specs['cflags'])

                            flags["ROUTINE_CLASS"] = class_name
                            flags["PRECISION_{}".format(precision.upper())] = None


                            if api=="blas":
                                flags["TEST_BLAS_API"] = None
                                routine_dict = self.parse_flags(routine, routine_flags)
                                flags.update(routine_dict)
                                routine_testclass_src = "blasapi_"+routine_class["testclass_src"]
                                routine_name = "{precision}{routine}".format(precision=precision[0], routine=class_name)
                            else:
                                routine_testclass_src = routine_class["testclass_src"]
                                routine_name = "{precision}{routine}".format(precision=precision[0], routine=routine)

                            flags["ROUTINE_CLASS_C"] = join(TESTCLASSES_PATH, routine_testclass_src)
                            flags["ROUTINE"] = routine_name
                            flags["ROUTINE_FULLNAME"] = routine

                            # add blas_api flag arguments values

                            self.recipe["routines"][routine] = {
                                "group": group_name,
                                "class": class_name,
                                "make_cmd": "update",
                                "flags": flags
                            }

        if scheduled_routines:
            print("Some routines were not found in the schema ({}) {}"
                  .format(RECIPE_SCHEMA, scheduled_routines))

    def run_all_recipes(self):
        # tune the recipe and run

        for la in self.specs["las"]:
            print("Testing {la}".format(la=la))
            self.recipe["cflags"]["LA"]=la

            if la=="REFERENCE":
                self.run_recipe()
                break

            if la=="BLAS_WRAPPER":
                self.run_recipe()
                break

            for target in self.specs["targets"]:
                print("Testing {target}".format(target=target))

                self.recipe["cflags"]["TARGET"]=target
                self.run_recipe()

    def run_recipe(self):
        # preparation step
        cflags = self.recipe["cflags"]
        #  make_flags = self.recipe["make_flags"]
        env_flags = self.recipe["env_flags"]

        #  if not self.VERBOSE: make_flags.update({"-s":None})

        _build_libblasfeo = cflags.get("BUILD_LIBS")
        _deploy_libblasfeo = cflags.get("DEPLOY_LIBS")


        if self.cli_flags.rebuild:
            _build_libblasfeo = 1
            _deploy_libblasfeo = 1

        if _build_libblasfeo:
            # compile also BLAS_API by default
            libblasfeo_flags = dict(cflags)
            libblasfeo_flags.update({"BLAS_API":1})
            make("-C .. ", libblasfeo_flags, env_flags)
        if _deploy_libblasfeo:
            make("-C .. deploy_to_tests", cflags, env_flags)

        for routine_name, args in self.recipe['routines'].items():
            # update local flags with global flags

            if args.get("flags"): args["flags"].update(cflags)
            else: args["flags"] = cflags

            if args.get("env_flags"): args["env_flags"].update(env_flags)
            else: args["env_flags"] = env_flags

            status =  self.test_routine(routine_name, args)

    def test_routine(self, routine_name, args):

        cflags = args["flags"]
        env_flags = args["env_flags"]

        print("\nTesting {}:{}\n".format(cflags['TARGET'], routine_name))

        status = make_templated(args["make_cmd"], cflags, env_flags)

        if not status:
            print("Error with {}:{} ({}/{})".format(cflags['TARGET'], routine_name, self.DONE, self.TOTAL))
            return status

        self.DONE += 1
        print("\nTested {}:{} ({}/{})".format(cflags['TARGET'], routine_name, self.DONE, self.TOTAL))

        return status



if __name__ == "__main__":

    cli_flags = parse_arguments()

    # generate recipes
    # test set to be run in the given excution of the script
    cookbook = CookBook(cli_flags)
    #  print(json.dumps(cookbook.recipe, indent=4))
    cookbook.run_all_recipes()
