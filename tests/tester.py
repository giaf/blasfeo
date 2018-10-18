#! /usr/bin/python

import subprocess
import sys
import json
import argparse

RECIPE_SCHEMA="test_schema.json"
RECIPE_JSON="recipe_default.json"

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



def make(cmd="", make_flags={}, env_flags={}):

    make_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in make_flags.items()])
    env_flags = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])

    run_cmd = "{env_flags} make {make_flags} {cmd}".format(env_flags=env_flags, make_flags=make_flags, cmd=cmd)

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
    def __init__(self, cli_flags):

        self.cli_flags=cli_flags

        with open(cli_flags.recipe_json) as f:
            self.specs = json.load(f)

        with open(RECIPE_SCHEMA) as f:
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

                            routine_name = "{}{}".format(precision, available_routine)
                            routine_fullname = routine_name
                            flags["ROUTINE"] = routine_fullname
                            flags["ROUTINE_CLASS"] = routine_subclass

                            self.recipe["routines"][routine_name] = {
                                "class": routine_class,
                                "subclass": routine_subclass,
                                "make_cmd": "update_{}{}".format(precision, routine_class),
                                "flags": flags
                            }

        if scheduled_routines:
            print("Some routines not found in the schema {}".format(scheduled_routines))

    def run_all_recipes(self):
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

    def run_recipe(self):
        # preparation step
        make_flags = self.recipe["make_flags"]
        env_flags = self.recipe["env_flags"]

        _silent = ""
        if not self.VERBOSE: _silent="-s"

        _build_libblasfeo = make_flags.get("BUILD_LIBS")
        _deploy_libblasfeo = make_flags.get("DEPLOY_LIBS")

        if self.cli_flags.rebuild:
            _build_libblasfeo = 1
            _deploy_libblasfeo = 1

        if _build_libblasfeo:
            make("{} -C .. ".format(_silent), make_flags, env_flags)
        if _deploy_libblasfeo:
            make("{} -C .. deploy_to_tests".format(_silent), make_flags, env_flags)

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

        print("\nTesting {}:{}\n".format(make_flags['TARGET'], routine_name))

        status = make(args["make_cmd"], make_flags, env_flags)

        if not status:
            print("Error with {}:{} ({}/{})".format(make_flags['TARGET'], routine_name, self.DONE, self.TOTAL))
            return status

        self.DONE += 1
        print("\nTested {}:{} ({}/{})".format(make_flags['TARGET'], routine_name, self.DONE, self.TOTAL))

        return status



if __name__ == "__main__":

    cli_flags = parse_arguments()

    # generate recipes
    # test set to be run in the given excution of the script
    cookbook = CookBook(cli_flags)
    #  print(json.dumps(cookbook.recipe, indent=4))
    cookbook.run_all_recipes()
