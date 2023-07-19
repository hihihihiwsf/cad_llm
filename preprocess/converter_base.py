class ConverterBase():
    """
    Base class to log and count conversion failures
    """
    def __init__(self):

        # Count of how many sketch conversions have been attempted
        self.count = 0
        # Count of how many successful (non fatal) conversions
        self.converted_count = 0

        # Count of the total sketch constraints
        self.constraint_count = 0
        # Count of the converted sketch constraints
        self.constraint_converted_count = 0

        # Count of the total sketch dimensions
        self.dimension_count = 0
        # Count of the converted sketch dimensions
        self.dimension_converted_count = 0

        # Count of the number of perfect sketch conversions without dropping constraints
        self.perfect_sketch_converted_count = 0

        # Log of the failures
        self.failures = {}
    
    def log_failure(self, failure):
        """Log a failure as either a fatal Exception or non fatal string"""
        fatal = False
        failure_str = failure
        if isinstance(failure, Exception):
            fatal = True
            failure_str = f"{type(failure).__name__} - {failure.args[0]}"
        
        if failure_str in self.failures:
            self.failures[failure_str]["count"] += 1
        else:
            self.failures[failure_str] = {
                "count": 1,
                "fatal": fatal
            }
    
    def print_log_results(self):
        """Print log result output"""
        print("---------------------")
        print("Fatal Failure Log")
        sorted_failures = dict(sorted(self.failures.items(), key=lambda item: item[1]["count"], reverse=True))
        for failure, failure_dict in sorted_failures.items():
            if failure_dict["fatal"]:
                print(f" - {failure}: {failure_dict['count']}")
        print("---------------------")
        print("Failure Log")
        for failure, failure_dict in sorted_failures.items():
            if not failure_dict["fatal"]:
                print(f" - {failure}: {failure_dict['count']}")
        print("---------------------")  
        print("Conversion Stats")
        constraint_converted_percentage = (self.constraint_converted_count / self.constraint_count) * 100
        print(f" - {self.constraint_converted_count}/{self.constraint_count} ({constraint_converted_percentage:.2f}%) constraints converted")
        dimension_converted_percentage = (self.dimension_converted_count / self.dimension_count) * 100
        print(f" - {self.dimension_converted_count}/{self.dimension_count} ({dimension_converted_percentage:.2f}%) dimensions converted")
        perfect_sketch_percentage = (self.perfect_sketch_converted_count / self.converted_count) * 100
        print(f" - {self.perfect_sketch_converted_count}/{self.converted_count} ({perfect_sketch_percentage:.2f}%) sketches converted without removing constraints")
        print("---------------------")
