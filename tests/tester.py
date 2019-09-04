#! /usr/bin/python

import subprocess as sp
import jinja2 as jn
import sys
import json
import argparse
import re
from hashlib import sha1
from pathlib import Path
from collections import OrderedDict
import shutil

BLASFEO_PATH=str(Path(__file__).absolute().parents[1])
BLASFEO_TEST_PATH=str(Path(__file__).absolute().parents[0])

TEST_SCHEMA="test_schema.json"
RECIPE_JSON="recipe_default.json"
BUILDS_DIR="build"
REPORTS_DIR="reports"
TESTCLASSES_DIR="classes"
TPL_PATH="Makefile.tpl"
LIB_BLASFEO_STATIC = "libblasfeo.a"
LIB_BLASFEO_REF_STATIC = "libblasfeo_ref.a"
MAKE_FLAGS={}
SILENT=0

def parse_arguments():
	parser = argparse.ArgumentParser(description='BLAFEO tests scheduler')

	parser.add_argument(dest='recipe_json', type=str, default=RECIPE_JSON, nargs='?',
		help='Run a batch of test from a specific recipe, i.e. recipe_all.json')
	parser.add_argument('--silent', default=False, action='store_true',
		help='Silent makefile output')
	parser.add_argument('--continue', dest='continue_test', default=False, action='store_true',
		help='Do not interrupt tests sweep on error')
	parser.add_argument('--rebuild', default=False, action='store_true',
		help='Rebuild libblasfeo to take into account recent code '+
		'changes or addition of new target to the recipe batch')

	args = parser.parse_args()
	return args


def make_blasfeo(cmd="", env_flags={}, blasfeo_flags={}):

	blasfeo_flags_str = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in blasfeo_flags.items()])
	env_flags_str = " ".join(["{k}={v}".format(k=k, v=v) for k, v in env_flags.items()])
	make_flags_str = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in MAKE_FLAGS.items()])

	run_cmd = "make clean"
	run_cmd = "{env_flags} make static_library -j 8 {blasfeo_flags} {make_flags} -C .. {cmd}"\
		.format(env_flags=env_flags_str, blasfeo_flags=blasfeo_flags_str, make_flags=make_flags_str, cmd=cmd)

	if not SILENT: print(run_cmd,"\n")

	make = sp.Popen(run_cmd,
		shell=True,
		stdin=sp.PIPE,
		stdout=sp.PIPE,
		stderr=sp.PIPE
		)

	outs, errs = make.communicate()

	if outs and not SILENT: print("Make infos:\n{}".format(outs.decode("utf8")))
	if errs: print("Make errors:\n{}".format(errs.decode("utf8")))

	return make.returncode


def make_templated(make_cmd="", env_flags={}, test_macros={}, blasfeo_flags={}, **kargs):

	# compress run_id
	la=blasfeo_flags['LA']
	if la != "HIGH_PERFORMANCE": target=la

	run_id = "{target}_{routine_fullname}_kstack{kstack}"\
		.format(
			target=blasfeo_flags['TARGET'],
			routine_fullname=kargs["fullname"],
			kstack=blasfeo_flags['K_MAX_STACK']
			)

	print("\nTesting {run_id}\n".format(run_id=run_id))

	with open(TPL_PATH) as f:
		template = jn.Template(f.read())

	makefile = template.render(test_macros=test_macros)

	# flag to override default libblasfeo flags
	blasfeo_flags_cmd = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in blasfeo_flags.items()])
	make_flags = " ".join(["{k}={v}".format(k=k, v=v) if v is not None else k for k, v in MAKE_FLAGS.items()])

	run_cmd = "make {blasfeo_flags_cmd} -f - {make_cmd}".format(make_cmd=make_cmd, blasfeo_flags_cmd=blasfeo_flags_cmd)
	report_cmd = "make {blasfeo_flags_cmd} {make_flags} {make_cmd}"\
		.format(make_cmd=make_cmd, make_flags=make_flags, blasfeo_flags_cmd=blasfeo_flags_cmd)

	if not SILENT: print(run_cmd,"\n")

	make = sp.Popen(run_cmd.split(),
		stdin=sp.PIPE,
		stdout=sp.PIPE,
		stderr=sp.PIPE
		)

	outs, errs = make.communicate(makefile.encode("utf8"))

	if outs and not SILENT: print("Make infos:\n{}".format(outs.decode("utf8")))
	if errs: print("Make errors:\n{}".format(errs.decode("utf8")))

	if make.returncode:
		# write report
		report_path = Path(REPORTS_DIR, run_id)
		report_path.mkdir(parents=True, exist_ok=True)
		with open(str(Path(report_path, "Makefile")), "w") as f:
			f.write(makefile)
		with open(str(Path(report_path, "make.sh")), "w") as f:
			f.write("#! /bin/bash\n")
			f.write(report_cmd+"\n")

		print("Error with {run_id}".format(run_id=run_id))

	else:
		if not SILENT: print("Tested with {run_id}".format(run_id=run_id))

	return make.returncode

class CookBook:
	def __init__(self, cli_flags):
		global SILENT

		self.cli_flags=cli_flags
		self.continue_test = 0

		with open(cli_flags.recipe_json) as f:
			self.specs = json.load(f, object_pairs_hook=OrderedDict)


		if self.specs["options"].get("silent") or self.cli_flags.silent:
			SILENT = 1

		if self.specs["options"].get("continue") or  self.cli_flags.continue_test:
			self.continue_test = 1

		with open(TEST_SCHEMA) as f:
			self.schema = json.load(f, object_pairs_hook=OrderedDict)

		self._success_n = 0
		self._errors_n = 0

		self._total_n =\
			len(set(self.specs["routines"]))\
			* len(self.specs["TARGET"])\
			* len(self.specs["K_MAX_STACK"])\
			* len(self.specs["precisions"])\
			* len(self.specs["apis"])

		# build standard recipe skelethon
		self.build_recipe()

	def parse_routine_options(self, routine_name, available_flags):

		# routine without any flag
		if not available_flags:
			return {'routine_basename': routine_name}

		pattern = '(?P<routine_basename>[a-z]*)_'

		for flag_name, flags_values in available_flags.items():
			flags_values = '|'.join(flags_values)
			pattern += '(?P<{flag_name}>[{flags_values}])'.format(flag_name=flag_name, flags_values=flags_values)

		parsed_flags = re.search(pattern, routine_name)

		if not parsed_flags:
			print("Error parsing flags of routine: {routine_name}".format(routine_name=routine_name))
			return {}

		return parsed_flags.groupdict()

	def build_recipe(self):
		scheduled_routines = set(self.specs['routines'])

		# create recipe with no global flags
		self.recipe = OrderedDict(self.specs)
		self.recipe["scheduled_routines"] = {}

		available_groups = self.schema['routines']

		# routine groups: blas1 blas2 ..
		for group_name, available_classes in available_groups.items():

			# routine classes: gemm, trsm, ...
			for class_name, routine_class in available_classes.items():
				available_routines = routine_class["routines"]
				routine_flags = routine_class["flags"]

				# routines: gemm, gemm_nn, gemm_nt, ...
				for routine  in available_routines:

					if routine not in scheduled_routines:
						continue

					scheduled_routines = scheduled_routines - {routine}

					# precision
					for precision in self.specs["precisions"]:

						# apis
						for api in self.specs["apis"]:

							test_macros = {}

							test_macros["ROUTINE_CLASS"] = class_name
							test_macros["PRECISION_{}".format(precision.upper())] = None

							routine_fullname = "{api}_{precision}{routine}".format(
								api=api, precision=precision[0], routine=routine)

							if api=="blas":
								test_macros["TEST_BLAS_API"] = None
								routine_dict = self.parse_routine_options(routine, routine_flags)
								if not routine_dict: continue
								test_macros.update(routine_dict)
								routine_testclass_src = "blasapi_"+routine_class["testclass_src"]
								routine_name = "{precision}{routine}".format(precision=precision[0], routine=class_name)
							else:
								routine_testclass_src = routine_class["testclass_src"]
								routine_name = "{precision}{routine}".format(precision=precision[0], routine=routine)

							test_macros["ROUTINE_CLASS_C"] = str(Path(TESTCLASSES_DIR, routine_testclass_src))
							test_macros["ROUTINE"] = routine_name
							test_macros["ROUTINE_FULLNAME"] = routine

							# add blas_api flag arguments values

							self.recipe["scheduled_routines"][routine_fullname] = {
								"group": group_name,
								"class": class_name,
								"api": api,
								"precision": precision,
								"make_cmd": "update",
								"fullname": routine_fullname,
								"test_macros": test_macros
							}

		if scheduled_routines:
			print("Some routines were not found in the schema ({}) {}"
				  .format(TEST_SCHEMA, scheduled_routines))

	def run_all(self):
		# tune the recipe and run

		for la in self.specs["LA"]:
			self.recipe["blasfeo_flags"]["LA"]=la

			if la=="REFERENCE":
				self.run_recipe()
				break

			if la=="EXTERNAL_BLAS_WRAPPER":
				self.run_recipe()
				break


			for target in self.specs["TARGET"]:
				self.recipe["blasfeo_flags"]["TARGET"]=target

				for max_stack in self.specs["K_MAX_STACK"]:
					self.recipe["blasfeo_flags"]["K_MAX_STACK"]=max_stack
					print("\n## Testing {la}:{target} kswitch={max_stack}".format(target=target, la=la, max_stack=max_stack))

					self.run_recipe()

	def is_lib_updated(self):

		target = self.recipe["blasfeo_flags"]["TARGET"]
		la = self.recipe["blasfeo_flags"]["LA"]

		if self.lib_static_dst.is_file() and self.libref_static_dst.is_file():
			return 1

		return 0

	def run_recipe(self):
		# preparation step
		test_macros = self.recipe["test_macros"]
		blasfeo_flags = self.recipe["blasfeo_flags"]
		env_flags = self.recipe["env_flags"]

		# always compile blas api
		blasfeo_flags.update({"BLAS_API":1})
		blasfeo_flags.update({"BLASFEO_PATH":str(BLASFEO_PATH)})

		blasfeo_flags_str = "_".join(
			["{k}={v}".format(k=k, v=v) if v is not None else '' for k, v in blasfeo_flags.items()])

		m = sha1(blasfeo_flags_str.encode("utf8"))
		binary_dir = m.hexdigest()
		binary_path = Path(BLASFEO_TEST_PATH, BUILDS_DIR, binary_dir)
		# create directory
		binary_path.mkdir(parents=True, exist_ok=True)
		# update binary_dir flag
		blasfeo_flags.update({"ABS_BINARY_PATH":str(binary_path)})

		self.lib_static_src = Path(BLASFEO_PATH, "lib", LIB_BLASFEO_STATIC)
		self.libref_static_src = Path(BLASFEO_PATH, "lib", LIB_BLASFEO_REF_STATIC)

		self.lib_static_dst = Path(binary_path, LIB_BLASFEO_STATIC)
		self.libref_static_dst = Path(binary_path, LIB_BLASFEO_REF_STATIC)

		lib_flags_json = str(Path(binary_path, "flags.json"))

		if self.cli_flags.silent or self.recipe["options"].get("silent"):
			MAKE_FLAGS.update({"-s":None})

		if self.cli_flags.rebuild or self.recipe["options"].get("rebuild") or not self.is_lib_updated():
			# compile the library
			make_blasfeo(blasfeo_flags=blasfeo_flags, env_flags=env_flags)

			# write used flags
			with open(lib_flags_json, "w") as f:
				json.dump(blasfeo_flags, f, indent=4)

			# copy library
			shutil.copyfile(str(self.lib_static_src), str(self.lib_static_dst))
			shutil.copyfile(str(self.libref_static_src), str(self.libref_static_dst))

			#  self.lib_static_dst.write_bytes(lib_static_src.read_bytes())
			#  self.libref_static_dst.write_bytes(libref_static_src.read_bytes())

		for routine_name, args in self.recipe['scheduled_routines'].items():
			# update local flags with global flags

			if args.get("test_macros"):
				args["test_macros"].update(test_macros)
			else:
				args["test_macros"] = test_macros

			if self.continue_test:
				args["test_macros"].update({"CONTINUE_ON_ERROR":1})

			if args.get("env_flags"):
				args["env_flags"].update(env_flags)
			else:
				args["env_flags"] = env_flags

			if args.get("blasfeo_flags"):
				args["blasfeo_flags"].update(blasfeo_flags)
			else:
				args["blasfeo_flags"] = blasfeo_flags

			error =  self.test_routine(routine_name, args)

			if error and not self.continue_test:
				break

	def test_routine(self, routine_fullname, kargs):

		error = make_templated(**kargs)

		if not error:
			self._success_n += 1
		else:
			self._errors_n += 1

		print("({done}:Succeded, {errors}:Errors) / ({total}:Total)"
			.format(done=self._success_n, errors=self._errors_n, total=self._total_n))

		return error



if __name__ == "__main__":

	cli_flags = parse_arguments()

	# generate recipes
	# test set to be run in the given excution of the script
	cookbook = CookBook(cli_flags)
	#  print(json.dumps(cookbook.recipe, indent=4))
	cookbook.run_all()
