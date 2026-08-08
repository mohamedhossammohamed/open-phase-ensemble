name: Feature Request
description: Suggest an idea or new detector module for open-phase-ensemble
labels: [enhancement]
body:
  - type: textarea
    id: summary
    attributes:
      label: Feature Proposal
      description: What feature or new detector algorithm would you like to see?
    validations:
      required: true
  - type: textarea
    id: motivation
    attributes:
      label: Motivation & Theoretical Rationale
      description: Why is this feature or detector useful for non-parametric time-series analysis?
    validations:
      required: true
