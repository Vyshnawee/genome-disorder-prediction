import sys


class AGDPMException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message
        self.error_detail = error_detail

    def __str__(self):
        return str(self.error_message)