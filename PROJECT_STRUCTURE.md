# Project Structure

- `code project.py`  
  Initial dataset loading and baseline text-classification pipeline example.

- `code project lime.py`  
  Initial LIME-based example script for text-classification explanations.

- `bios_bias_explainability_report.py`  
  End-to-end experiment script that:
  - trains a profession classifier on `LabHC/bias_in_bios`
  - performs gender-swap counterfactual bias checks
  - compares attention, SHAP, and LIME explanations
  - exports report-ready tables/figures and a markdown report

- `report_outputs/` *(generated when the experiment runs)*  
  Stores exported tables, figures, and summary report artifacts.
