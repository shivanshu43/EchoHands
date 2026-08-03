class VariationManager:
    
    # Guides the user through different hand
    # variations while collecting the dataset
   

    def __init__(self):

        self.variations = [

            "Center",

            "Move Hand Left",

            "Move Hand Right",

            "Move Hand Up",

            "Move Hand Down",

            "Move Closer",

            "Move Further",

            "Rotate Wrist Left",

            "Rotate Wrist Right",

            "Tilt Palm"

        ]

        self.current_index = 0

    def get_current_variation(self):

        return self.variations[self.current_index]

    def next_variation(self):

        if self.current_index < len(self.variations) - 1:

            self.current_index += 1

    def reset(self):

        self.current_index = 0