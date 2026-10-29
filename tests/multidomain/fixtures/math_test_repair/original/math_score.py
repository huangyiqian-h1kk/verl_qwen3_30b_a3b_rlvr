"""Bounded scoring: unresolved timeouts return explicit zero; system faults raise."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from . import POLICY
from .backend import verify_archive, VERSIONS

ATTEMPTS = ((5,20),(15,60),(30,120))

class ScoringFailure(RuntimeError):
    pass


def check_versions():
    verify_archive()
    result = _process({'action':'versions'},20)
    if result.get('status') != 'ok' or result.get('versions') != VERSIONS:
        raise ScoringFailure('Isolated verifier preflight failed: '+repr(result))
    return result['versions']


def normalize_gold(expected):
    expected = expected.strip()
    if expected.startswith('\\(') and expected.endswith('\\)'):
        expected = expected[2:-2].strip()
    if expected.startswith('$') and expected.endswith('$') and len(expected)>1:
        expected = expected[1:-1].strip()
    return '\\boxed{'+expected+'}'


def _process(payload,budget):
    verify_archive()
    cmd = [sys.executable,'-I',str(Path(__file__).with_name('backend_worker.py'))]
    started = time.monotonic()
    process = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    timed_out = False
    try:
        try:
            output, error = process.communicate(json.dumps(payload,ensure_ascii=False),timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            output,error = process.communicate(timeout=5)
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        phases = [r['phase'] for r in rows if r.get('status')=='phase']
        phase = phases[-1] if phases else 'imports'
        final = [r for r in rows if r.get('status')!='phase']
        if timed_out:
            result = dict(status='timeout' if phase in ('gold_parse','prediction_parse','comparison') else 'error',
                          phase=phase,error=f'Verifier process exceeded {budget}s')
        elif process.returncode != 0 or len(final)!=1:
            result = dict(status='error',phase=phase,error=f'Verifier process/protocol failure rc={process.returncode}',stderr=error[-4000:])
        else:
            result = final[0]
        return dict(result,elapsed_seconds=time.monotonic()-started)
    except (ValueError,KeyError,TypeError) as exc:
        raise ScoringFailure('Malformed verifier protocol') from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _attempt(expected,generated,seconds,budget):
    return dict(_process(dict(expected=expected,generated=generated,seconds=seconds),budget),
                operation_timeout_seconds=seconds,process_timeout_seconds=budget)


def score_only(expected,generated,seconds=5,phase=lambda name:None):
    result = _attempt(expected,generated,seconds,max(20,seconds*4))
    if result['status']!='ok':
        raise ScoringFailure(repr(result))
    return result['score']


def _save(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _score(expected,generated,diagnostics_dir,sample_id=None,details=False):
    record = dict(policy=POLICY,sample_id=str(sample_id) if sample_id is not None else None,
        expected_answer=expected,generated_text=generated,
        input_sha256=hashlib.sha256(json.dumps([expected,generated],ensure_ascii=False).encode()).hexdigest(),
        host=socket.gethostname(),pid=os.getpid(),pbs_job_id=os.environ.get('PBS_JOBID'),
        started_utc=datetime.now(timezone.utc).isoformat(),attempts=[])
    path = Path(diagnostics_dir)/(record['input_sha256'][:16]+'_'+uuid.uuid4().hex+'.json')
    for number,(seconds,budget) in enumerate(ATTEMPTS,1):
        result = _attempt(expected,generated,seconds,budget)
        record['attempts'].append(result)
        if result['status']=='ok':
            value = result.get('score')
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value not in (0,1):
                result.update(status='error',error='Invalid verifier reward')
            else:
                outcome = dict(score=float(value),verifier_timeout=0.0,
                               verifier_repaired=float(bool(result.get('repaired'))),verifier_retry_count=float(number-1))
                if number>1 or result.get('repaired'):
                    record.update(status='RECOVERED' if number>1 else 'EXACT_CHECKED',resolved=True,**outcome)
                    _save(path,record)
                return outcome if details else float(value)
        if result['status']=='timeout' and result.get('phase') in ('prediction_parse','gold_parse','comparison'):
            if number<len(ATTEMPTS):
                record.update(status='RETRYING',resolved=False)
                _save(path,record)
                continue
            outcome = dict(score=0.0,verifier_timeout=1.0,verifier_repaired=0.0,verifier_retry_count=float(number-1))
            record.update(status='TIMEOUT_FALLBACK',resolved=False,reason='verifier_timeout',**outcome)
            _save(path,record)
            print(f'[0390] verifier_timeout: unresolved sample={sample_id}; fallback reward=0; diagnostic={path}',flush=True)
            return outcome if details else 0.0
        record.update(status='INFRASTRUCTURE_ERROR',resolved=False)
        _save(path,record)
        raise ScoringFailure(f"Verifier infrastructure failure: {result.get('error')}; diagnostic={path}")
    raise AssertionError('Unreachable')


def math_score_details(expected,generated,timeout=10,*,diagnostics_dir,sample_id=None):
    try:
        return _score(expected,generated,diagnostics_dir,sample_id,details=True)
    except (ScoringFailure,OSError,RuntimeError) as exc:
        from multidomain.common import InfrastructureError
        raise InfrastructureError(str(exc)) from exc


def math_score(expected,generated,timeout=10,*,diagnostics_dir,sample_id=None):
    return math_score_details(expected,generated,timeout,diagnostics_dir=diagnostics_dir,sample_id=sample_id)['score']
