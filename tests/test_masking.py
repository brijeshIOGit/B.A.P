"""R1: one test per mask, plus numbers that must NOT be masked."""

from codeforge.masking import luhn_ok, mask, mask_code


def test_openai_style_key_is_masked():
    clean, found = mask("my key is sk-proj-AbCdEfGhIjKlMnOp1234")
    assert clean == "my key is [SECRET]" and found == ["secret"]


def test_groq_key_is_masked():
    assert mask("gsk_ABCDEFGHIJKLMNOPQRSTUV12")[0] == "[SECRET]"


def test_xai_key_is_masked():
    assert mask("xai-ABCDEFGHIJKLMNOPQRSTUV12")[0] == "[SECRET]"


def test_email_is_masked():
    assert mask("mail priya@example.com now")[0] == "mail [EMAIL] now"


def test_valid_card_is_masked():
    assert mask("card 4111 1111 1111 1111 please")[0] == "card [CARD] please"


def test_aadhaar_with_spaces_is_masked():
    assert mask("aadhaar 2345 6789 0123")[0] == "aadhaar [AADHAAR]"


def test_aadhaar_without_spaces_is_masked():
    assert mask("aadhaar:234567890123")[0] == "aadhaar:[AADHAAR]"


def test_indian_phone_is_masked():
    assert mask("call +91 98765 43210")[0] == "call [PHONE]"


def test_order_number_is_not_masked():
    assert mask("where is order 12345678?")[0] == "where is order 12345678?"


def test_card_like_number_failing_luhn_is_not_masked():
    assert mask("ref 1234567890123456")[0] == "ref 1234567890123456"


def test_small_numbers_in_code_are_not_masked():
    text = "assert fibonacci(30) == 832040 and year == 2024"
    assert mask(text)[0] == text


def test_word_sk_learn_is_not_a_secret():
    assert mask("use sk-learn for this")[0] == "use sk-learn for this"


def test_masking_twice_changes_nothing():
    once = mask("sk-proj-AbCdEfGhIjKlMnOp1234 and 2345 6789 0123")[0]
    assert mask(once)[0] == once


def test_code_keeps_example_emails_but_masks_secrets():
    code = 'EMAIL = "user@example.com"\nKEY = "sk-proj-AbCdEfGhIjKlMnOp1234"\n'
    out = mask_code(code)
    assert "user@example.com" in out and "[SECRET]" in out


def test_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("1234567890123456")
