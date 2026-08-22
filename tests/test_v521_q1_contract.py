from crcv_q1.protocol import Q1Protocol,hash_protocol
from crcv_q1.split_guard import audit_rows


def test_v521_q1_identity_and_floors_are_not_weakened():
    p=Q1Protocol().validate()
    assert p.version=="5.21-q1-v2" and p.proposed_method=="crcv_v521"
    assert p.mean_dice_gain_floor==.010 and p.mean_crack_iou_gain_floor==.005
    assert p.min_backbones>=5 and p.min_datasets>=3 and len(hash_protocol(p))==64


def test_final_split_requires_hash_and_exact_duplicate_fails():
    good=[
      {"sample_id":"a","split":"fit","lineage_id":"l1","source_dataset":"D","historically_exposed":"true","image_sha256":"a"*64},
      {"sample_id":"b","split":"final_external","lineage_id":"l2","source_dataset":"E","historically_exposed":"false","image_sha256":"b"*64},
    ]
    assert audit_rows(good)["status"]=="PASS"
    nohash=[dict(x) for x in good]; nohash[1].pop("image_sha256")
    assert audit_rows(nohash)["status"]=="FAIL"
    dup=[dict(x) for x in good]; dup[1]["image_sha256"]="a"*64
    assert audit_rows(dup)["status"]=="FAIL"


def test_same_split_exact_duplicate_is_rejected():
    rows=[
      {"sample_id":"a","split":"final_external","lineage_id":"l1","source_dataset":"E","historically_exposed":"false","image_sha256":"a"*64},
      {"sample_id":"b","split":"final_external","lineage_id":"l2","source_dataset":"E","historically_exposed":"false","image_sha256":"a"*64},
    ]
    r=audit_rows(rows)
    assert r["status"]=="FAIL" and any("exact duplicate image" in x for x in r["failures"])


def test_invalid_image_sha_is_rejected():
    rows=[{"sample_id":"a","split":"final_external","lineage_id":"l1","source_dataset":"E","historically_exposed":"false","image_sha256":"not-a-sha"}]
    assert audit_rows(rows)["status"]=="FAIL"
