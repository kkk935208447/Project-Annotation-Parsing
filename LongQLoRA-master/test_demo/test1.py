import sys
from loguru import logger
# 重定义终端显示logger颜色
logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<cyan><lvl>{level:8}</></>| {name} : {module}:{line:4} | <cyan>mymodule</> | - <lvl>{message}</>",
        "colorize": True
    },
])
 
logger.debug('this is debug')
logger.info('this is info')
logger.success('this is success')
logger.warning('this is warning')
logger.error('this is error')
logger.critical('this is critical')
