"""Pidfd operations for Python builds that omit the optional Linux wrappers.

Only the Python binding is replaced. Kernel errors/permissions still propagate;
there is no fallback to an unbound numeric-PID kill or process-group signal.
"""
import ctypes
import os
import platform
import signal
import sys


def _syscall(number, *arguments):
    # Linux x86-64 and AArch64 LP64 use these same UAPI numbers:
    # https://github.com/torvalds/linux/blob/v6.6/arch/x86/entry/syscalls/syscall_64.tbl
    # https://github.com/torvalds/linux/blob/v6.6/include/uapi/asm-generic/unistd.h
    if (sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8
            or platform.machine().lower() not in ("x86_64", "aarch64")):
        raise RuntimeError("Native pidfd compatibility supports Linux x86_64/aarch64 LP64 only")
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.syscall
    call.restype = ctypes.c_long
    call.argtypes = [ctypes.c_long]  # Remaining arguments are explicitly typed varargs.
    ctypes.set_errno(0)
    result = call(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, "pidfd syscall: " + os.strerror(error))
    return result


def open_process(pid):
    native = getattr(os, "pidfd_open", None)
    if callable(native):
        return native(pid, 0)
    # Kernel returns a close-on-exec descriptor; flags=0 works before Linux 5.10.
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_signal(descriptor, signum):
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        return native(descriptor, signum, None, 0)
    return _syscall(424, ctypes.c_int(descriptor), ctypes.c_int(signum),
                    ctypes.c_void_p(None), ctypes.c_uint(0))


def probe():
    """Read-only permission/capability check on this process, before any stop."""
    descriptor = open_process(os.getpid())
    try:
        send_signal(descriptor, 0)  # Signal 0 checks access; it delivers no signal.
    finally:
        os.close(descriptor)
    return dict(status="PASS", machine=platform.machine(), kernel=platform.release(),
                open_backend="python" if callable(getattr(os, "pidfd_open", None)) else "linux-syscall",
                signal_backend="python" if callable(getattr(signal, "pidfd_send_signal", None)) else "linux-syscall")
