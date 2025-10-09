import sys, sagemaker
print("Python:", sys.executable)
print("sagemaker version:", sagemaker.__version__)

import sagemaker.automl.automlv2 as m
print("Has TabularJobConfig:", hasattr(m, "TabularJobConfig"))
print("Available names with 'Tabular':", [n for n in dir(m) if "Tabular" in n])