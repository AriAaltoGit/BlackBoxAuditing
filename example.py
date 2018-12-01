# import BlackBoxAuditing
import BlackBoxAuditing as BBA
# import machine learning technique
from BlackBoxAuditing.model_factories import SVM, DecisionTree, NeuralNetwork

"""
Using a preloaded dataset
"""
# load in preloaded dataset
#data = BBA.load_data("german")
data = BBA.load_data("adult")

# initialize the auditor and set parameters
auditor = BBA.Auditor()
#auditor.ModelFactory= SVM
auditor.ModelFactory= SVM

# call the auditor with the data
#auditor(data, output_dir="german_audit_output", dump_all=False)
auditor(data, output_dir="adult_audit_output")

# find contexts of discrimination in dataset
#auditor.find_contexts("age_cat", output_dir="german_context_output")
#auditor.find_contexts("age", output_dir="adult_context_output")