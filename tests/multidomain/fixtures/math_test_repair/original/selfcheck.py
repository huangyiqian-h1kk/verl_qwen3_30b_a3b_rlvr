"""CPU-only checks against the REAL pinned verifier, including real spawn."""
import json
import tempfile
from .math_score import check_versions, normalize_gold, score_only, _score

CASES = (
    ("integer_correct", "42", r"The answer is \boxed{42}.", 1.0),
    ("integer_incorrect", "42", r"\boxed{41}", 0.0),
    ("empty", "42", "", 0.0),
    ("fraction", r"\frac{1}{2}", r"\boxed{0.5}", 1.0),
    ("fraction_wrong", r"\frac{1}{2}", r"\boxed{0.25}", 0.0),
    ("radical", r"\sqrt{2}", r"\boxed{\sqrt{2}}", 1.0),
    ("gold_dollars", "$7$", r"\boxed{7}", 1.0),
    ("gold_parens", r"\(7\)", r"\boxed{7}", 1.0),
    ("symbolic", "x+1", r"\boxed{1+x}", 1.0),
    ("symbolic_wrong", "x+1", r"\boxed{x+2}", 0.0),
    ("malformed_prediction", "42", r"\boxed{\frac{}}", 0.0),
)


GAMMA_GOLD = r"\frac{(e^{\pi}-e^{-\pi})\Gamma(2015+i)\Gamma(2015-i)}{4\pi}"


def main():
    versions = check_versions()
    checks = list(CASES) + [
        ('real_gamma_product', GAMMA_GOLD, r'\boxed{\prod_{k=2}^{2014}(k^2+1)}',1.0),
        ('wrong_gamma_product_limit', GAMMA_GOLD, r'\boxed{\prod_{k=2}^{2013}(k^2+1)}',0.0),
    ]
    with tempfile.TemporaryDirectory(prefix='0390_db_math_cpu_') as temp:
        for name,gold,pred,expected in checks:
            actual = _score(gold,pred,temp,name,details=True)
            if actual['score'] != expected or actual['verifier_timeout']:
                raise RuntimeError('CPU scoring mismatch/unresolved: '+name+' '+repr(actual))
    print('[0390] CPU math PASS:',json.dumps(dict(versions=versions,real_subprocess_cases=len(checks),
        models_loaded=False,gpu_jobs_submitted=0)),flush=True)


if __name__ == "__main__":
    main()
