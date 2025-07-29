import sys
import logging
from src.logger import logging

def error_message_detail(error, error_detail: sys): # type: ignore
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename # type: ignore
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] with message: [{str(error)}]" # type: ignore
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys): # type: ignore
        super().__init__(error_message_detail(error, error_detail))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message


        