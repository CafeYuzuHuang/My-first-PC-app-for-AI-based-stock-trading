# -*- coding: utf-8 -*-

import os
from functools import wraps
import logging
from collections import namedtuple

"""
myLog.py的功能：
(1) 客製化的日誌功能
(2) 錯誤(或警告)訊息的處理

Levels
=====================================================
安全等級        數值          說明            輸出函數
=====================================================
logging.NOTSET    0        未設定                  x
logging.DEBUG    10      除錯等級     logging.debug()
logging.INFO     20      訊息等級      logging.info()
logging.WARNING  30      警告等級   logging.warning()
logging.ERROR    40      錯誤等級     logging.error()
logging.CRITICAL 50  嚴重錯誤等級  logging.critical()
=====================================================
"""

# global variables and constants:
LogFolder = '' # folder to place .log files
if __name__ == "__main__":
    LogFolder = os.path.dirname(os.getcwd()) + "\\log"
else:
    modulepath = os.path.dirname(__file__)
    LogFolder = os.path.dirname(modulepath) + "\\log"
# print(LogFolder)
if not os.path.isdir(LogFolder):
    os.mkdir(LogFolder)

LogFmt = '%(asctime)s\t - %(levelname)s - \t%(name)s: %(message)s'
DateFmt = '%Y/%m/%d %H:%M:%S'
LogFileMode = 'a' # default mode
LogExt = '.log'
RootLogName = 'root'

# Utilities:
def ArgList(*args, **kwargs):
    """
    A utility function for decorator applications.
    Called by Log_Func() to show arguments.
    """
    s = ""
    for a in args:
        s += repr(a) + ", "
    for k,v in kwargs.items():
        s += "{k} = {v}, ".format(k= k, v=repr(v))
    return "({s})".format(s=s)

def GetDest(name):
    """
    A utility function returns absolute path of a log file.
    If name is None, return the root logger.
    """
    if not os.path.isdir(LogFolder):
        os.mkdir(LogFolder)
    logf = RootLogName
    if name != None:
        logf = str(name)
    return LogFolder + '\\' + logf + LogExt

# Functions and classes of myLog:
def SetRootLogger(level = logging.DEBUG):
    """ A convenient way to set root logger. """
    dest = GetDest(None)
    logger = logging.getLogger()
    logger.handlers.clear() # 避免多次執行getLogger後Handler重複產生
    logger.setLevel(level)
    logging.basicConfig(filename = dest,
                        filemode = LogFileMode,
                        format = LogFmt,
                        datefmt = DateFmt)
    return logger

def CreateLogger(name = None, level = logging.DEBUG):
    """
    A function returns a logger object, applied to decorators as well.
    The created one may be either root or childen logger.
    """
    dest = GetDest(name)
    logger = logging.getLogger(name = name)
    logger.handlers.clear() # 避免多次執行getLogger後Handler重複產生
    logger.setLevel(level)
    fh = logging.FileHandler(dest, encoding = 'utf-8')
    formatter = logging.Formatter(LogFmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def DeleteLogFiles(flist = None):
    """ Remove .log file(s) in the log folder. """
    logfs = os.listdir(LogFolder)
    if flist is not None:
        logfs = flist # override by user-specified file(s)
        if type(logfs) != list:
            logfs = [logfs]
        print("\n*** List specified: ***\n%s\nStarts to clean log file(s):" % logfs)
    else:
        print("\n*** List gathered: ***\n%s\nStarts to clean log file(s):" % logfs)
    for logf in logfs:
        fpath = LogFolder + '\\' + logf
        try:
            fext = os.path.splitext(fpath)[-1]
            if fext.lower() == LogExt: # check file extension
                fs = os.path.getsize(fpath) # file size in bytes
                os.remove(fpath)
                print("%s (size: %d bytes) is removed!" % (logf, fs))
            else:
                print("%s is not a log file (not removed)." % logf)
        except:
            print("%s cannot be removed!" % logf)
            pass
    print("*** DeleteLogFile finishes! ***\n")
    return None

def DisableAllLoggers():
    """ Disable all loggers """
    logging.disable(level = logging.CRITICAL)
    return None

def EnableAllLoggers():
    """ (Re)enable all loggers """
    logging.disable(level = logging.NOTSET)
    return None

# A named tuple for logging levels
Levels = namedtuple('LogLevel', \
                    ['NOTSET', 'DEBUG', 'INFO', \
                     'WARNING', 'ERROR', 'CRITICAL'])
LogLevel = Levels(logging.NOTSET, logging.DEBUG,
                  logging.INFO, logging.WARNING,
                  logging.ERROR, logging.CRITICAL)

class Log(logging.getLoggerClass()):
    def __init__(self, level = logging.DEBUG, name = None):
        self.level = level
        self.name = name
        if self.name is None:
            self.logger = SetRootLogger(self.level)
        else:
            self.logger = CreateLogger(self.name, self.level)
        self.logger.propagate = False # not pass to ancestor logger(s)
        # self.logger.propagate = True # default value
    def debug(self, msg): # Applied to logging anything
        self.logger.debug(msg)
    def info(self, msg):
        self.logger.info(msg)
    def warning(self, msg):
        self.logger.warning(msg)
    def error(self, msg):
        self.logger.error(msg)
    def critical(self, msg): # Applied to close logging
        self.logger.critical(msg)
    def log(self, level, msg):
        self.logger.log(level, msg)
    def setLevel(self, level):
        self.logger.setLevel(level)

# Additional: decorators
def Log_Error(level = logging.ERROR):
    """ A decorator for exception message logging """
    def err_log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = CreateLogger("Err", level)
                err_msg = "An error has occurred at " + func.__name__
                logger.exception(err_msg)
                return e
        return wrapper
    return err_log

def Log_Func(level = logging.DEBUG, name = None, msg = None):
    """ A decorator for function message logging. """
    def fun_log(func):
        logname = name if name else func.__module__
        logmsg = msg if msg else logname + '.' + func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = CreateLogger(logmsg, level)
            call_msg = logmsg + ArgList(*args, **kwargs)
            logger.log(level, call_msg)
            return_val =  func(*args, **kwargs)
            return_msg = logmsg + " returns {r}".format(r=return_val)
            logger.log(level, return_msg)
            return return_val
        return wrapper
    return fun_log


"""
# Function testing
# Prepare two functions with decorators
@Log_Error(logging.ERROR)
@Log_Func(logging.DEBUG, None, None)
def Div(num, dom): # an example
    return num/dom

@Log_Error(logging.ERROR)
@Log_Func(logging.DEBUG, None, None)
def Sleep_Sqrt(num): # test Log_Error
    from time import sleep
    from math import sqrt
    sleep(1)
    return sqrt(num)

if __name__ == "__main__":
    
    print("Start logging test:")
    print("Show log folder: ", LogFolder)
    # 用自訂類別建立logger，並寫入訊息
    Log1 = Log(LogLevel.INFO, None)
    Log1.info("Logging from Log1 without name. (should be root logger)")
    Name2 = str(__name__).strip('_')
    Log2 = Log(LogLevel.INFO, Name2)
    Log2.info("Logging from Log2 with name %s" % Name2)
    Log2.debug("Logging a lower-level message (should be NOT logged).")
    Log2.error("Logging a higher-level message (should be logged).")
    # 切換logger
    Log1.info("Logging from Log1 again.")
    Log2.info("Logging from Log2 again.")    
    print("Finish logging test!")
    
    # 上述測試成功後，再進行以下測試：用logging建立裝飾器
    print("Start decorator test:")
    Div(2, 99) # pass
    Div(2.0, 99.0) # pass
    Sleep_Sqrt(2.0) # pass
    Sleep_Sqrt(-1) # raise exception
    Div(2.0, 0.0) # raise exception
    Div(2.0, 'aa') # raise exception
    Sleep_Sqrt('aa') # raise exception
    print("Finish decorator test:")
    
# Done!
"""
