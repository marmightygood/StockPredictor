import datetime
import os

class logger(object):
    """My awesome logger"""

    file = None

    def __init__(self, log_dir, log_name):
        file = open(log_dir + log_name + "_" + str(os.getpid()) + ".log", "a")

    def write(log_message):
        try: 
            formatted_message = datetime.datetime.now().isoformat(' ') + ": " + log_message
            file.write(formatted_message)
        except IOError as e:
            print ("IOError writing to {0}\n".format(log_dir + log_name + "_" + str(os.getpid()) + ".log"))

lm = logger("C:\\temp\\", "test_log")
lm.write (log_message = "")