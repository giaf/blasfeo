#! /usr/bin/python

import subprocess
import sys
import json


class CookBook:
    def __init__(self, recipe_specs="batch_run.json", recipe_schema="test_schema.json"):

        with open(recipe_specs) as f:
            self.specs = json.load(f)
        with open(recipe_schema) as f:
            self.schema = json.load(f)

        self.DONE = 0
        self.SILENT = 0

        self.TOTAL =\
            len(self.specs["routines"])\
            * len(self.specs["targets"])\
            * len(self.specs["precisions"])

        # build standard recipe skelethon
        self.build_recipe()


    def run_all_recipes(self):
        # tune the recipe and run

        for la in self.specs["las"]:
            print(f"Testing {la}")
            self.recipe["flags"]["LA"]=la

            if la=="REFERENCE":
                self.run_recipe()
                break

            if la=="BLAS_WRAPPER":
                self.run_recipe()
                break

            for target in self.specs["targets"]:
                print(f"Testing {target}")

                self.recipe["flags"]["TARGET"]=target
                self.run_recipe()


    def run_recipe(self):
        # preparation step
        flags = self.recipe["flags"]

        if flags["BUILD_LIBS"]:
            make(flags=flags, make_flags="-C ..")
        if flags["BUILD_LIBS"] and flags["DEPLOY_LIBS"]:
            make("deploy_to_tests", flags=flags, make_flags="-C ..")

        for routine_name, args in self.recipe['routines'].items():
            # update local flags with global flags
            if args.get("flags"): args["flags"].update(flags)
            else: args["flags"] = {}
            status =  self.test_routine(routine_name, args)

    def test_routine(self, routine_name, args):

        make_flags = ""
        if self.SILENT: make_flags += " -s"
        flags = args["flags"]

        print(f"\nTesting {flags['TARGET']}:{routine_name} ({self.DONE}/{self.TOTAL})\n")

        status = make(args["make_cmd"], flags, make_flags)

        if not status:
            print(f"Error with {flags['TARGET']}:{routine_name} ({self.DONE}/{self.TOTAL})")

        self.DONE += 1

        return status


    def build_recipe(self):
        scheduled_routines = set(self.specs['routines'])

        # create recipe with no global flags
        self.recipe = {'routines':{}, "flags":self.specs['global_flags']}

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
                            flags.update(self.specs['flags'])

                            routine_name = f"{precision}{available_routine}"
                            routine_fullname = f"blasfeo_{routine_name}"
                            flags["ROUTINE"] = routine_fullname
                            flags["ROUTINE_SUBCLASS"] = routine_subclass

                            self.recipe["routines"][routine_name] = {
                                "class": routine_class,
                                "subclass": routine_subclass,
                                "make_cmd": f"update_{precision}{routine_class}",
                                "flags": flags
                            }

        if scheduled_routines:
            print(f"Some routines not found in the schema {scheduled_routines}")



def make(cmd="", flags={}, make_flags=""):

    flags = " ".join([f"{k}={v}" for k, v in flags.items()])

    run_cmd = f"make {make_flags} {flags} {cmd}"
    print(run_cmd)
    make_process = subprocess.Popen(run_cmd,
        shell=True, stdout=subprocess.PIPE, stderr=sys.stdout.fileno())

    while True:
        line = make_process.stdout.readline()
        if not line: break
        print(line.decode("utf-8").strip())
        sys.stdout.flush()
    #  return fail
    return 1



if __name__ == "__main__":

    # generate recipes
    # test set to be run in the given excution of the script
    cookbook = CookBook()
    #  print(json.dumps(cookbook.recipe, indent=4))
    cookbook.run_all_recipes()
