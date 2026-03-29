def test_import():
    import gpt_oss_rs

    assert hasattr(gpt_oss_rs, "Sampler")
    assert hasattr(gpt_oss_rs, "Tokenizer")
    assert gpt_oss_rs.__version__ == "0.1.0"
