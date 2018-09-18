#! /usr/bin/python

import subprocess
import sys
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='BLAFEO tests scheduler')

    parser.add_argument('--run', dest='batch_run', type=str, default="batch_run.json",
                        help='Batch run json file')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level')

    args = parser.parse_args()
    return args



def make(cmd="", make_flags={}, env_flags={}):

    make_flags = " ".join([f"{k}={v}" for k, v in make_flags.items()])
    env_flags = " ".join([f"{k}={v}" for k, v in env_flags.items()])

    run_cmd = f"{env_flags} make {make_flags} {cmd}"

    print(run_cmd)
    make_process = subprocess.Popen(run_cmd,
        shell=True, stdout=subprocess.PIPE, stderr=sys.stdout.fileno())

    while True:
        line = make_process.stdout.readline()
        if not line: break
        print(line.decode("utf-8").strip())
        sys.stdout.flush()
    return 1


class CookBook:
    def __init__(self,
        cli_flags,
        recipe_specs="batch_run.json",
        recipe_schema="test_schema.json"):

        self.cli_flags=cli_flags

        with open(recipe_specs) as f:
            self.specs = json.load(f)
        with open(recipe_schema) as f:
            self.schema = json.load(f)

        self.DONE = 0

        self.VERBOSE = self.cli_flags.verbose

        self.TOTAL =\
            len(self.specs["routines"])\
            * len(self.specs["targets"])\
            * len(self.specs["precisions"])

        # build standard recipe skelethon
        self.build_recipe()

    def build_recipe(self):
        scheduled_routines = set(self.specs['routines'])

        # create recipe with no global flags
        self.recipe = {
            'routines':{},
            'make_flags':self.specs['make_flags'],
            'env_flags':self.specs['env_flags']
        }


        available_classes = self.schema['routines']

        for routine_class in available_classes:
            available_subclasses = available_classes[routine_class]

            for routine_subclass in available_subclasses:
                available_routines = available_subclasses[routine_subclass]

                for available_routine in available_routines:

                    if available_routine in scheduled_routines:
                        scheduled_routines = scheduled_routines - {available_routine}

                        for precision in self.specs["precisions"]:

                            flags = {}
                            flags.update(self.specs['make_flags'])

                            routine_name = f"{precision}{available_routine}"
                            routine_fullname = f"blasfeo_{routine_name}"
                            flags["ROUTINE"] = routine_fullname
                            flags["ROUTINE_CLASS"] = routine_subclass

                            self.recipe["routines"][routine_name] = {
                                "class": routine_class,
                                "subclass": routine_subclass,
                                "make_cmd": f"update_{precision}{routine_class}",
                                "flags": flags
                            }

        if scheduled_routines:
            print(f"Some routines not found in the schema {scheduled_routines}")

    def run_all_recipes(self):
        # tune the recipe and run

        for la in self.specs["las"]:
            print(f"Testing {la}")
            self.recipe["make_flags"]["LA"]=la

            if la=="REFERENCE":
                self.run_recipe()
                break

            if la=="BLAS_WRAPPER":
                self.run_recipe()
                break

            for target in self.specs["targets"]:
                print(f"Testing {target}")

                self.recipe["make_flags"]["TARGET"]=target
                self.run_recipe()

    def run_recipe(self):
        # preparation step
        make_flags = self.recipe["make_flags"]
        env_flags = self.recipe["env_flags"]

        _silent = ""
        if not self.VERBOSE: _silent="-s"

        if make_flags["BUILD_LIBS"]:
            make(f"{_silent} -C .. ", make_flags, env_flags)

        if make_flags["BUILD_LIBS"] and make_flags["DEPLOY_LIBS"]:
            make(f"{_silent} -C .. deploy_to_tests", make_flags, env_flags)

        for routine_name, args in self.recipe['routines'].items():
            # update local flags with global flags

            if args.get("flags"): args["flags"].update(make_flags)
            else: args["flags"] = make_flags

            if args.get("env_flags"): args["env_flags"].update(env_flags)
            else: args["env_flags"] = env_flags

            status =  self.test_routine(routine_name, args)

    def test_routine(self, routine_name, args):

        make_flags = args["flags"]
        env_flags = args["env_flags"]

        print(f"\nTesting {make_flags['TARGET']}:{routine_name}\n")

        status = make(args["make_cmd"], make_flags, env_flags)

        if not status:
            print(f"Error with {make_flags['TARGET']}:{routine_name} ({self.DONE}/{self.TOTAL})")
            return status

        self.DONE += 1
        print(f"\nTested {make_flags['TARGET']}:{routine_name} ({self.DONE}/{self.TOTAL})\n")

        return status



if __name__ == "__main__":

    cli_flags = parse_arguments()

    # generate recipes
    # test set to be run in the given excution of the script
    cookbook = CookBook(cli_flags)
    #  print(json.dumps(cookbook.recipe, indent=4))
    cookbook.run_all_recipes()
