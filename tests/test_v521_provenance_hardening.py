import hashlib
from pathlib import Path
from crcv_q1.evidence import validate_artifact_ref, validate_run_record


def test_malformed_hex_digests_are_rejected(tmp_path):
    p=tmp_path/'x'; p.write_text('x')
    ref={'path':'x','sha256':'z'*64,'kind':'test'}
    assert any('invalid sha256' in x for x in validate_artifact_ref(ref,tmp_path))
    rec={'experiment_id':'e','git_commit':'nothex!!','dataset_manifest_sha256':'g'*64,'split_manifest_sha256':'b'*64,'config_sha256':'c'*64,'base_artifact_sha256':'d'*64,'probability_provenance_bound':True,'seed':1,'backbone':'u','dataset':'d','resolution':128,'method':'crcv_v521','artifacts':[{'path':'x','sha256':hashlib.sha256(b'x').hexdigest(),'kind':'test'}]}
    failures=validate_run_record(rec,tmp_path)
    assert any('commit' in x for x in failures) and any('dataset_manifest' in x for x in failures)


def test_run_record_requires_base_probability_binding(tmp_path):
    p=tmp_path/'x'; p.write_text('x'); good_sha=hashlib.sha256(b'x').hexdigest()
    rec={'experiment_id':'e','git_commit':'abcdef1','dataset_manifest_sha256':'a'*64,'split_manifest_sha256':'b'*64,'config_sha256':'c'*64,'base_artifact_sha256':'d'*64,'probability_provenance_bound':False,'seed':1,'backbone':'u','dataset':'d','resolution':128,'method':'crcv_v521','artifacts':[{'path':'x','sha256':good_sha,'kind':'test'}]}
    failures=validate_run_record(rec,tmp_path)
    assert any('probability_provenance_bound' in x for x in failures)
