"""An isolated, pinned, pure-Python verifier; no training-site imports."""
from functools import lru_cache
import hashlib
from pathlib import Path

VERSIONS = {'math-verify':'0.9.0', 'latex2sympy2_extended':'1.11.0',
            'sympy':'1.14.0', 'mpmath':'1.3.0', 'antlr4-python3-runtime':'4.13.2'}
ARCHIVE_SHA256 = '067bb4557182f7b3159006dbaa11b449b617b175784a15e84b7895f65aea8393'


@lru_cache(maxsize=1)
def verify_archive():
    path = Path(__file__).with_name('verifier_backend.zip')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError('Isolated verifier archive differs from the pinned build')
    return path


def activate():
    import sys
    if sys.version_info[:2] != (3,12):
        raise RuntimeError('This isolated verifier bundle is pinned to training Python 3.12')
    path = verify_archive()
    # Executed only in python -I backend_worker.py, never in a Ray/torch process.
    sys.path.insert(0,str(path))
    from importlib.metadata import version
    actual = {name:version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise RuntimeError('Wrong isolated verifier versions: '+repr(actual))
    for name in ('math_verify','latex2sympy2_extended','sympy','mpmath','antlr4'):
        module = __import__(name)
        if not str(module.__file__).startswith(str(path)+'/'):
            raise RuntimeError('Verifier import escaped isolated archive: '+name)
    return actual
