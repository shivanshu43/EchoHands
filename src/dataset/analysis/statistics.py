class DatasetStatistics:
    
    # Keeps track of dataset collection progress
    

    def __init__(self, target_samples):

        self.target_samples = target_samples
        self.sample_count = 0

    def increment(self):

        self.sample_count += 1

    def get_count(self):

        return self.sample_count

    def get_remaining(self):

        return self.target_samples - self.sample_count

    def is_complete(self):

        return self.sample_count >= self.target_samples

    def get_progress(self):

        if self.target_samples == 0:
         return 0

        return (
            self.sample_count / self.target_samples
        ) * 100