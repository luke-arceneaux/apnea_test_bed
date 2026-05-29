import numpy as np
from csv import DictWriter
np.set_printoptions(precision=3)

runs_schema = [
    "time","subject_id","signal","session","num_expert_annot","num_nst_annot","num_matched","num_missed",
    "num_false_pos","percent_matched","percent_missed","percent_false_pos","accuracy","precision","recall","f1","sensitivity","expert_AHI","expert_AHI_severity",
    "nst_AHI","nst_AHI_severity","total_hypoxia_load","tst90","plot","include_apnea","include_hypopnea",
    "result","reason","stage_reached", "schema_version"
]

''' Defines Logger class to log metadata, scores, metrics, ... '''
# class Logger:
#     def __init__(self):
#         self.has_meta = False
#         # metadata 
#         self.meta = {}
#         self.counts = {}
#         self.metrics = {}
#         self.scores = {}

#         self.all_annot = {}  # contains expert annots, and matched nst annots
#         self.extra_pos_annot = {} # contains nst false pos annots
#         self.expert_annots = {}
#         self.nst_annots = {}
#         self.result = {}
#         self.reason = {}

        
#     def log_meta(self, k, v):
#         self.meta[k] = v

#     def log_metrics(self, k, v):
#         self.metrics[k] = round(v,3)

#     def log_scores(self, k, v):
#         self.scores[k] = v
    
#     def write_row(self, out_file, d):
#         row = {k: d.get(k, None) for k in runs_schema}

#         # with open(out_file, 'a', newline='\n') as f_object:
#         #     dictwriter_object = DictWriter(f_object, fieldnames=list(d.keys()))
#         #     dictwriter_object.writerow(d)
#         #     f_object.close()

#         with open(out_file, 'a', newline='\n') as f_object:
#             dictwriter_object = DictWriter(f_object, fieldnames=runs_schema)
#             dictwriter_object.writerow(row)
#             f_object.close()

from datetime import datetime
import traceback

class Logger:
    def __init__(self):
        self.has_meta = False

        self.meta = {}
        self.counts = {}
        self.metrics = {}
        self.scores = {}

        self.all_annot = {}
        self.extra_pos_annot = {}
        self.expert_annots = {}
        self.nst_annots = {}

        self.result = {}
        self.reason = {}

        # NEW
        self.stage_reached = None
        self.failed = False
        self.error_type = None
        self.error_message = None
        self.traceback = None
        self.start_time = datetime.utcnow()
        self.end_time = None

    def mark_stage(self, stage):
        self.stage_reached = stage

    def fail(self, exc: Exception):
        self.failed = True
        self.result = "FAILED"
        self.reason = str(exc)
        self.error_type = type(exc).__name__
        self.traceback = traceback.format_exc()
        self.end_time = datetime.utcnow()

    def succeed(self):
        self.result = "SUCCESS"
        self.end_time = datetime.utcnow()


# from dataclasses import dataclass, field
# from datetime import datetime
# import traceback

# @dataclass
# class RunContext:
#     run_id: str
#     s3_key: str

#     status: str = "STARTED"
#     start_time: datetime = field(default_factory=datetime.utcnow)
#     end_time: datetime | None = None

#     stage: str | None = None
#     error_type: str | None = None
#     error_message: str | None = None
#     traceback: str | None = None

#     # domain-specific fields
#     wav_duration: float | None = None
#     num_events: int | None = None
#     model_version: str | None = None

#     def mark_stage(self, stage: str):
#         self.stage = stage

#     def fail(self, exc: Exception):
#         self.status = "FAILED"
#         self.end_time = datetime.utcnow()
#         self.error_type = type(exc).__name__
#         self.error_message = str(exc)
#         self.traceback = traceback.format_exc()

#     def succeed(self):
#         self.status = "SUCCESS"
#         self.end_time = datetime.utcnow()


def write_run_row(logger, out_file="runs.csv"):
    row = {}

    row.update(logger.meta)
    row.update(logger.counts)
    row.update(logger.metrics)
    row.update(logger.scores)

    row["result"] = logger.result
    row["reason"] = logger.reason
    row["stage_reached"] = logger.stage_reached
    row["schema_version"] = "v3"

    final_row = {k: row.get(k, None) for k in runs_schema}

    with open(out_file, "a", newline="\n") as f:
        DictWriter(f, fieldnames=runs_schema).writerow(final_row)

