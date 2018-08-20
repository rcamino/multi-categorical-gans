import csv
import os
import time


class Logger(object):

    PRINT_FORMAT = "epoch {:d}/{:d} {}-{}: {:.05f} Time: {:.2f} s"
    CSV_COLUMNS = ["epoch", "model", "metric_name", "metric_value", "time"]

    start_time = None

    def __init__(self, output_path, append=False):
        if append and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            self.output_file = open(output_path, "a")
            self.output_writer = csv.DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
        else:
            self.output_file = open(output_path, "w")
            self.output_writer = csv.DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
            self.output_writer.writeheader()

        self.start_timer()

    def start_timer(self):
        self.start_time = time.time()

    def log(self, epoch_index, num_epochs, model_name, metric_name, metric_value):
        elapsed_time = time.time() - self.start_time

        self.output_writer.writerow({
            "epoch": epoch_index + 1,
            "model": model_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "time": elapsed_time
        })

        print(self.PRINT_FORMAT
              .format(epoch_index + 1,
                      num_epochs,
                      model_name,
                      metric_name,
                      metric_value,
                      elapsed_time
                      ))

    def flush(self):
        self.output_file.flush()

    def close(self):
        self.output_file.close()

        self.output_file = None
        self.output_writer = None
