name: Bug Report
description: Create a report to help us fix a bug or unexpected behavior
labels: [bug]
body:
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear and concise description of what the bug is.
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: Code snippet or steps to reproduce the issue.
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment Info
      description: OS version, Python version, package versions.
    validations:
      required: false
