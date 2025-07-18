---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Description**
A clear and concise description of what the bug is.

**Triton Information**
What version of Triton are you using?

Are you using the Triton container or did you build it yourself?

**To Reproduce**

If the problem appears to be a bug in the execution of the model itself, first attempt to run the model directly in ONNX Runtime. What is the output from loading and running the model in ORT directly? If there is a problem running the model directly with ORT, please submit an issue in the microsoft/onnxruntime (github.com) project.

If the problem appears to be in Triton itself, provide detailed steps to reproduce the behavior in Triton.

Describe the models (framework, inputs, outputs), ideally include the model configuration file (if using an ensemble include the model configuration file for that as well).

**Expected behavior**
A clear and concise description of what you expected to happen.
