name: Scientific Review & Independent Replication Request
description: Request or submit an independent peer review of benchmark results and math proofs
labels: [scientific-review, peer-review]
body:
  - type: textarea
    id: benchmark_target
    attributes:
      label: Dataset / Module Being Reviewed
      description: Which dataset (PhysioNet, CWRU, synthetic) or module math proof are you reviewing?
    validations:
      required: true
  - type: textarea
    id: methodology
    attributes:
      label: Replication Methodology & Results
      description: Provide independent VUS-ROC / VUS-PR metrics, surrogate null comparison, and methodology details.
    validations:
      required: true
  - type: textarea
    id: findings
    attributes:
      label: Findings & Challenges
      description: Did your results confirm or challenge our reported preliminary metrics?
    validations:
      required: true
