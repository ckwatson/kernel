import sys
import os
from ..data.molecular_species import molecular_species
from ..data.reaction_mechanism_class import reaction_mechanism
from ..data.condition_class import condition
from ..data.reagent import reagent
from ..data.puzzle_class import puzzle
from ..data.solution_class import solution

def name(class_obj):
	return class_obj.__name__

# depends on JSON base class
for class_being_tested in [molecular_species, condition, reaction_mechanism, reagent, puzzle, solution]:
	system_output = sys.stdout # store stdout
	sys.stdout = open(os.getcwd() + "/testing_result_" + name(class_being_tested) + ".txt", "w") # pipe to file
	test_result = class_being_tested.test()
	sys.stdout.close() # close file
	sys.stdout = system_output #replace stdout

	if test_result:
		print("PASSED", name(class_being_tested), sep="    ")
	else:
		print("FAILED", name(class_being_tested), sep="    ")

