from perturbfm.utils.hashing import sha256_json


def test_hash_is_stable_for_dict_order():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert sha256_json(a) == sha256_json(b)


def test_hash_changes_with_content():
    a = {"a": 1}
    b = {"a": 2}
    assert sha256_json(a) != sha256_json(b)
