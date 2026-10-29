"""One score per fresh subprocess; stdout is a JSON-lines protocol."""
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from d4_db_mathfix.backend import activate


def emit(row):
    print(json.dumps(row,ensure_ascii=False),flush=True)


def score(expected, generated, seconds, phase):
    import itertools
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
    from d4_db_mathfix.math_score import normalize_gold
    from d4_db_mathfix.exact_math import compare
    def extract(text, config):
        try:
            return parse(text,config,parsing_timeout=seconds,raise_on_error=True)
        except TimeoutException:
            raise
        except Exception:
            return []  # Same ordinary parse fallback as upstream.
    phase('prediction_parse')
    pred = extract(generated,(ExprExtractionConfig(),LatexExtractionConfig()))
    phase('gold_parse')
    gold = extract(normalize_gold(expected),(LatexExtractionConfig(),))
    if not gold:
        raise ValueError('No gold target; scoring configuration/data needs attention')
    phase('comparison')
    repaired = False
    exact_disagreement = False
    for g,p in itertools.product(gold,pred):
        # Bound our exact path by the same alarm as upstream symbolic work.
        from math_verify.utils import timeout
        result, route = timeout(timeout_seconds=seconds)(compare)(g,p)
        if result is not None:
            repaired = True
            if result:
                return dict(status='ok',score=1.0,route=route,repaired=True)
            exact_disagreement = True
            continue
        # Raw-string fallbacks cannot override an exact numeric disagreement.
        if exact_disagreement and (isinstance(g,str) or isinstance(p,str)):
            continue
        try:
            if verify(g,p,6,timeout_seconds=seconds,raise_on_error=True):
                return dict(status='ok',score=1.0,route='math_verify',repaired=repaired)
        except TimeoutException:
            raise
        except Exception:
            continue
    return dict(status='ok',score=0.0,route='exact_disagreement' if exact_disagreement else 'math_verify',repaired=repaired)


def main():
    phase_name = 'imports'
    def phase(name):
        nonlocal phase_name
        phase_name = name
        emit(dict(status='phase',phase=name))
    try:
        activate()
        payload = json.loads(sys.stdin.read())
        from math_verify.errors import TimeoutException
        if payload.get('action') == 'versions':
            from d4_db_mathfix.backend import VERSIONS
            emit(dict(status='ok',versions=VERSIONS))
            return
        # Keep library chatter off the protocol, but emit phases to the original
        # stdout so the parent can diagnose an externally killed calculation.
        stdout = sys.stdout
        def report(name):
            with contextlib.redirect_stdout(stdout):
                phase(name)
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result = score(payload['expected'],payload['generated'],payload['seconds'],report)
        except TimeoutException as exc:
            result = dict(status='timeout',phase=phase_name,error=type(exc).__name__,traceback=traceback.format_exc()[-8000:])
        emit(result)
    except BaseException as exc:
        emit(dict(status='error',phase=phase_name,error=type(exc).__name__+': '+str(exc)[:2000],
                  traceback=traceback.format_exc()[-8000:]))


if __name__ == '__main__':
    main()
