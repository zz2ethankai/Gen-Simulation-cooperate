import logging
import os
import time
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for older Python builds
    ZoneInfo = None

from nimbus.utils.config import save_config


DEFAULT_LOG_TIMEZONE = "Asia/Shanghai"
VELOCITY_TRACE_LOGGER_NAME = "de_velocity_trace"


def _get_log_timezone():
    tz_name = os.environ.get("INTERNDATA_LOG_TZ", DEFAULT_LOG_TIMEZONE)
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    if tz_name in {"Asia/Shanghai", "PRC", "CST", "UTC+8", "+08:00", "+0800"}:
        return timezone(timedelta(hours=8), "CST")
    return datetime.now().astimezone().tzinfo


def _format_log_timestamp(log_tz):
    return datetime.now(log_tz).strftime("%Y%m%d_%H%M%S_%f")


def _configure_velocity_trace_logging(log_dir, timestamp, formatter):
    """Send high-volume motion traces to a dedicated file.

    The trace logger is deliberately non-propagating so per-step velocity
    records do not get copied into the regular data-engine log.  A tagged
    handler is replaced when logging is configured more than once in the
    same process (for example in a worker restart), while unrelated handlers
    remain untouched.
    """

    logger = logging.getLogger(VELOCITY_TRACE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if not getattr(handler, "_interndata_velocity_trace", False):
            continue
        logger.removeHandler(handler)
        handler.close()

    detail_file = os.path.join(log_dir, f"de_velocity_trace_{timestamp}.log")
    handler = logging.FileHandler(detail_file, mode="a")
    handler._interndata_velocity_trace = True
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class LocalTimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", log_tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.log_tz = log_tz or _get_log_timezone()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.log_tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def configure_logging(exp_name, name=None, config=None):
    pod_name = os.environ.get("POD_NAME", None)
    if pod_name is not None:
        exp_name = f"{exp_name}/{pod_name}"
    log_dir = os.path.join("./output", exp_name)
    log_tz = _get_log_timezone()
    timestamp = _format_log_timestamp(log_tz)
    if name is None:
        log_name = f"de_time_profile_{timestamp}.log"
    else:
        log_name = f"de_{name}_time_profile_{timestamp}.log"

    log_file = os.path.join(log_dir, log_name)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.makedirs(log_dir, exist_ok=True)
            break
        except Exception as e:
            print(f"Warning: Stale file handle when creating {log_dir}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            else:
                raise RuntimeError(f"Failed to create log directory {log_dir} after {max_retries} attempts") from e

    if config is not None:
        config_log_file = os.path.join(log_dir, "de_config.yaml")
        save_config(config, config_log_file)

    logger = logging.getLogger("de_logger")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="a")
    formatter = LocalTimezoneFormatter("%(asctime)s - %(levelname)s - %(message)s", log_tz=log_tz)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    _configure_velocity_trace_logging(log_dir, timestamp, formatter)
    logger.info("Start Data Engine")

    return logger
