from .target_alignment import TargetAlignmentJudge
from .reasoning_soundness import ReasoningSoundnessJudge
from .execution_accuracy import ExecutionAccuracyJudge
from .output_integrity import OutputIntegrityJudge

ALL_JUDGES = [
    TargetAlignmentJudge,
    ReasoningSoundnessJudge,
    ExecutionAccuracyJudge,
    OutputIntegrityJudge,
]
