import base64
import json
import re
import pytest
import importlib
import itertools

# Import the Flask app and helpers from index.py
index = importlib.import_module("index")
app = index.app

# ---------- Helpers used by tests ----------

def api_convert(client, input_value, input_type, output_type):
    """Call the /convert API and return (result, error)."""
    resp = client.post(
        "/convert",
        data=json.dumps({"input": input_value, "inputType": input_type, "outputType": output_type}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "result" in payload and "error" in payload
    return payload["result"], payload["error"]


def int_to_b64_le(n: int) -> str:
    """Little-endian minimal-length base64 encoding for a non-negative integer."""
    if n < 0:
        raise ValueError("Only non-negative integers are supported")
    if n == 0:
        b = b"\x00"
    else:
        b = n.to_bytes((n.bit_length() + 7) // 8, byteorder="little", signed=False)
    return base64.b64encode(b).decode("utf-8")


def b64_le_to_int(s: str) -> int:
    """Decode LE base64-encoded integer string to int."""
    b = base64.b64decode(s)
    return int.from_bytes(b, byteorder="little", signed=False)


def _norm_text(s: str) -> str:
    """Normalize number words to compare 'forty two' vs 'forty-two' etc."""
    return re.sub(r"[\s\-]+", " ", s.strip().lower())


# ---------- Smoke test for index route ----------

def test_index_route_renders():
    with app.test_client() as client:
        r = client.get("/")
        # App should render an HTML page (template), but we only care that it succeeds.
        assert r.status_code == 200


# ---------- Unit tests for helper functions that index.py exposes ----------

def test_text_to_number_valid_cases():
    # These should work with either the simple or robust parser
    assert index.text_to_number("zero") == 0
    assert index.text_to_number("nil") == 0
    assert index.text_to_number("One") == 1
    assert index.text_to_number("two") == 2
    assert index.text_to_number("ten") == 10


def test_text_to_number_invalid():
    # Use a word that's truly invalid regardless of parser breadth
    with pytest.raises(ValueError):
        index.text_to_number("spaghetti")


def test_number_to_text_small_numbers():
    # num2words for small values is consistent in English
    assert index.number_to_text(0).lower() == "zero"
    assert index.number_to_text(5).lower() == "five"
    assert index.number_to_text(10).lower() == "ten"


def test_base64_number_helpers_little_endian_spec():
    """Specification: integers in base64 should be little-endian by default.
    These values are the LE encodings (expected by the assignment spec).
    """
    cases = {
        0: "AA==",
        1: "AQ==",
        255: "/w==",
        256: "AAE=",
        513: "AQI=",
    }
    for n, expected in cases.items():
        assert int_to_b64_le(n) == expected, f"LE base64 for {n}"
        # And inverse
        assert b64_le_to_int(expected) == n


# ---------- /convert end-to-end tests for each input type to all outputs ----------

@pytest.mark.parametrize(
    "input_value,input_type,expected_decimal",
    [
        ("five", "text", 5),
        ("nil", "text", 0),
        ("101010", "binary", 42),
        ("52", "octal", 42),
        ("42", "decimal", 42),
        ("2a", "hexadecimal", 42),
        ("AQ==", "base64", 1),       # 1 in LE (and BE) is the same
        ("/w==", "base64", 255),     # 255 is the same in LE and BE (single byte)
        ("AAE=", "base64", 256),     # LE differs from BE for multi-byte values
    ],
)
def test_convert_to_decimal_from_various_inputs(input_value, input_type, expected_decimal):
    with app.test_client() as client:
        result, err = api_convert(client, input_value, input_type, "decimal")
        assert err is None
        assert int(result) == expected_decimal


@pytest.mark.parametrize(
    "number,expected_bin,expected_oct,expected_hex,expected_b64_le",
    [
        (0, "0", "0", "0", "AA=="),
        (1, "1", "1", "1", "AQ=="),
        (5, "101", "5", "5", "BQ=="),
        (10, "1010", "12", "a", "Cg=="),
        (42, "101010", "52", "2a", "Kg=="),
        (255, "11111111", "377", "ff", "/w=="),
        (256, "100000000", "400", "100", "AAE="),  # LE expectation
        (513, "1000000001", "1001", "201", "AQI="),  # LE expectation
    ],
)
def test_convert_from_decimal_to_all_outputs(number, expected_bin, expected_oct, expected_hex, expected_b64_le):
    with app.test_client() as client:
        # decimal -> binary
        result, err = api_convert(client, str(number), "decimal", "binary")
        assert err is None
        assert result == expected_bin

        # decimal -> octal
        result, err = api_convert(client, str(number), "decimal", "octal")
        assert err is None
        assert result == expected_oct

        # decimal -> hexadecimal
        result, err = api_convert(client, str(number), "decimal", "hexadecimal")
        assert err is None
        assert result == expected_hex

        # decimal -> base64 (LE per spec)
        result, err = api_convert(client, str(number), "decimal", "base64")
        assert err is None
        assert result == expected_b64_le, "Base64 should be little-endian by default"


@pytest.mark.parametrize("word,expected_le_b64", [("zero", "AA=="), ("one", "AQ=="), ("ten", "Cg==")])
def test_text_input_to_base64_and_back(word, expected_le_b64):
    with app.test_client() as client:
        # text -> base64
        result, err = api_convert(client, word, "text", "base64")
        assert err is None
        assert result == expected_le_b64

        # Then base64 -> decimal -> text (round-trip via API)
        dec, err = api_convert(client, result, "base64", "decimal")
        assert err is None
        text_back, err = api_convert(client, dec, "decimal", "text")
        assert err is None
        # Normalize casing for comparison
        assert _norm_text(text_back) == _norm_text(word)


@pytest.mark.parametrize(
    "b64,expected_decimal",
    [
        ("AA==", 0),
        ("AQ==", 1),
        ("/w==", 255),
        ("AAE=", 256),  # LE expectation (differs from BE)
        ("AQI=", 513),  # LE expectation (differs from BE)
    ],
)
def test_base64_input_little_endian_spec(b64, expected_decimal):
    with app.test_client() as client:
        result, err = api_convert(client, b64, "base64", "decimal")
        assert err is None
        assert int(result) == expected_decimal, "Base64 should decode using little-endian byte order"


def test_hex_input_accepts_prefix_and_case():
    with app.test_client() as client:
        for hx in ("0x2A", "2A", "2a"):
            result, err = api_convert(client, hx, "hexadecimal", "decimal")
            assert err is None
            assert int(result) == 42


def test_binary_octal_inputs_must_not_have_prefixes():
    with app.test_client() as client:
        # '0b' or '0o' prefixes should be invalid (int(x, base) rejects them)
        for inval, b in (("0b1010", "binary"), ("0o52", "octal")):
            result, err = api_convert(client, inval, b, "decimal")
            assert result is None
            assert err is not None


# ---------- Cross-conversion matrix to satisfy "all types" rubric ----------

def test_cross_conversion_matrix_for_all_types_42():
    # canonical representations for 42 across all supported types
    reps = {
        "text": "forty two",
        "binary": "101010",
        "octal": "52",
        "decimal": "42",
        "hexadecimal": "2a",
        "base64": "Kg==",  # 0x2a -> same for LE/BE (single byte)
    }
    types = list(reps.keys())

    with app.test_client() as client:
        for inp_t in types:
            for out_t in types:
                if inp_t == out_t:
                    # Skip identity to avoid coupling to specific identity behavior
                    continue
                inp = reps[inp_t]
                expected = reps[out_t]

                result, err = api_convert(client, inp, inp_t, out_t)
                assert err is None

                if out_t == "text":
                    assert _norm_text(result) == "forty two"
                elif out_t == "hexadecimal":
                    assert result.lower() == expected
                else:
                    assert result == expected


# ---------- Error handling: at least 3 distinct error classes ----------

def test_error_invalid_input_type():
    with app.test_client() as client:
        result, err = api_convert(client, "42", "roman", "decimal")
        assert result is None and err
        assert "invalid input type" in err.lower()


def test_error_invalid_output_type():
    with app.test_client() as client:
        result, err = api_convert(client, "42", "decimal", "quaternary")
        assert result is None and err
        assert "invalid output type" in err.lower()


@pytest.mark.parametrize("bad,kind", [
    ("102", "binary"),        # bad digit for base-2
    ("89", "octal"),          # bad digit for base-8
    ("xyz", "hexadecimal"),   # not hex
])
def test_error_malformed_numeric_for_base(bad, kind):
    with app.test_client() as client:
        result, err = api_convert(client, bad, kind, "decimal")
        assert result is None and err


def test_error_malformed_base64():
    with app.test_client() as client:
        result, err = api_convert(client, "***", "base64", "decimal")
        assert result is None and err


def test_error_unsupported_text_word():
    with app.test_client() as client:
        result, err = api_convert(client, "spaghetti", "text", "decimal")
        assert result is None and err


# ---------- Round-trip / property tests ----------

def test_round_trip_decimal_text_decimal_for_supported_range():
    """For numbers where text parsing is supported by index.text_to_number (0..10), round-trip should work."""
    with app.test_client() as client:
        for n in range(0, 11):
            text_value, err = api_convert(client, str(n), "decimal", "text")
            assert err is None
            back, err = api_convert(client, text_value, "text", "decimal")
            assert err is None
            assert int(back) == n


def test_round_trip_decimal_base64_decimal_property():
    """Round-trip through the app's own base64 encoder/decoder should preserve the integer.
    This test is endianness-agnostic and validates overall consistency.
    """
    with app.test_client() as client:
        for n in [0, 1, 42, 255, 256, 513, 65535, 65536, 2**20 + 3]:
            b64, err = api_convert(client, str(n), "decimal", "base64")
            assert err is None
            back, err = api_convert(client, b64, "base64", "decimal")
            assert err is None
            assert int(back) == n


def test_negative_decimal_to_base64_policy():
    """If negatives → base64 are disallowed, expect an error; if allowed, expect round-trip."""
    with app.test_client() as client:
        result, err = api_convert(client, "-1", "decimal", "base64")
        if err:
            assert result is None
        else:
            back, err2 = api_convert(client, result, "base64", "decimal")
            assert err2 is None
            assert int(back) == -1



def decimal_to_base64_little_endian(n: int) -> str:
    if n == 0:
        raw = b"\x00"
    else:
        byte_len = (n.bit_length() + 7) // 8
        raw = n.to_bytes(byte_len, byteorder="little", signed=False)
    return base64.b64encode(raw).decode("utf-8")

NUMERIC_TYPES = ["binary", "octal", "decimal", "hexadecimal", "base64"]
TEST_NUMBERS = [0, 1, 2, 7, 8, 15, 16, 42, 255, 256, 1024, 65535]

def encode_for_type(n: int, t: str) -> str:
    if t == "decimal":
        return str(n)
    if t == "binary":
        return bin(n)[2:]
    if t == "octal":
        return oct(n)[2:]
    if t == "hexadecimal":
        return hex(n)[2:]
    if t == "base64":
        return decimal_to_base64_little_endian(n)
    raise AssertionError(f"unknown type {t}")

def test_readme_decimal_to_binary_example():
    # "Convert decimal to binary: Input '42' decimal -> binary"
    with app.test_client() as client:
        result, err = api_convert(client, "42", "decimal", "binary")
        assert err is None
        assert result == "101010"

def test_readme_text_to_decimal_example_expected_42():
    # "Convert text to decimal: Input 'forty two' text -> decimal"
    # If the implementation cannot parse "forty two", this SHOULD FAIL,
    # surfacing a bug against the README spec.
    with app.test_client() as client:
        result, err = api_convert(client, "forty two", "text", "decimal")
        assert err is None, f"Expected to parse 'forty two' → 42 per README, got error: {err}"
        assert int(result) == 42

def test_readme_hex_to_text_roundtrip_via_decimal():
    # "Convert hexadecimal to text: Input '2a' hex -> text"
    # Avoid asserting exact hyphenation; just check it round-trips to 42.
    with app.test_client() as client:
        text_out, err = api_convert(client, "2a", "hexadecimal", "text")
        assert err is None
        back_dec, err2 = api_convert(client, text_out, "text", "decimal")
        assert err2 is None
        assert int(back_dec) == 42


# ---------- Base64 little-endian policy test ----------
@pytest.mark.parametrize("n,expected_b64", [(258, "AgE="), (1, "AQ=="), (0, "AA==")])
def test_base64_little_endian_expected_values(n, expected_b64):
    """Verify base64 uses LITTLE-ENDIAN byte order as required."""
    with app.test_client() as client:
        got, err = api_convert(client, str(n), "decimal", "base64")
        assert err is None
        assert got == expected_b64, f"Expected little-endian b64 for {n}"

        # And round-trip back to decimal
        back, err2 = api_convert(client, got, "base64", "decimal")
        assert err2 is None
        assert int(back) == n


# ---------- Full cross-product across numeric-only types ----------
NUMERIC_TYPES = ["binary", "octal", "decimal", "hexadecimal", "base64"]
TEST_NUMBERS = [0, 1, 2, 7, 8, 15, 16, 42, 255, 256, 1024, 65535]

def encode_for_type(n: int, t: str) -> str:
    if t == "decimal":
        return str(n)
    if t == "binary":
        return bin(n)[2:]
    if t == "octal":
        return oct(n)[2:]
    if t == "hexadecimal":
        return hex(n)[2:]
    if t == "base64":
        return decimal_to_base64_little_endian(n)
    raise AssertionError(f"unknown type {t}")

@pytest.mark.parametrize("n", TEST_NUMBERS)
@pytest.mark.parametrize("in_t,out_t", list(itertools.product(NUMERIC_TYPES, NUMERIC_TYPES)))
def test_all_numeric_type_conversions_roundtrip_via_decimal(n, in_t, out_t):
    """Exercise all reasonable flows across numeric-only types.

    Strategy: convert input string (typed) -> out_t, then convert back to decimal.
    The final decimal should equal the original n.
    """
    with app.test_client() as client:
        input_str = encode_for_type(n, in_t)
        out_str, err = api_convert(client, input_str, in_t, out_t)
        assert err is None, f"{in_t}->{out_t} failed for {n}: {err}"

        # Normalize by converting out_str back to decimal via API
        back_dec, err2 = api_convert(client, out_str, out_t, "decimal")
        assert err2 is None
        assert int(back_dec) == n


# ---------- Identity conversions ----------
@pytest.mark.parametrize("t", NUMERIC_TYPES + ["text"])
def test_identity_conversion_returns_equivalent_representation(t):
    """Converting a value from type t to type t should be equivalent (string-equal or round-trip equal)."""
    samples = {
        "decimal": "42",
        "binary": "101010",
        "octal": "52",
        "hexadecimal": "2a",
        "base64": "Kg==",  # 42 little-endian
        "text": "forty two",
    }
    val = samples[t]
    with app.test_client() as client:
        out, err = api_convert(client, val, t, t)
        assert err is None
        if t == "text":
            # Allow formatting differences by round-tripping via decimal
            back, err2 = api_convert(client, out, t, "decimal")
            assert err2 is None
            assert int(back) == 42
        else:
            assert out == val


# ---------- Error handling (at least 3 distinct errors) ----------
@pytest.mark.parametrize("bad_input,input_type", [
    ("10201", "binary"),     # invalid binary digit '2'
    ("89", "octal"),         # invalid octal digits 8/9
    ("xyz", "hexadecimal"),  # invalid hex
    ("***", "base64"),       # not valid base64
])
def test_invalid_digits_raise_errors(bad_input, input_type):
    with app.test_client() as client:
        out, err = api_convert(client, bad_input, input_type, "decimal")
        assert out is None
        assert isinstance(err, str) and err

def test_empty_input_is_error():
    with app.test_client() as client:
        out, err = api_convert(client, "", "decimal", "binary")
        assert out is None
        assert isinstance(err, str) and err

def test_unknown_input_type_is_error():
    with app.test_client() as client:
        out, err = api_convert(client, "42", "romannumeral", "decimal")
        assert out is None
        assert isinstance(err, str) and err

def test_missing_fields_is_error():
    with app.test_client() as client:
        # Missing outputType field
        resp = client.post("/convert", data=json.dumps({"input": "42", "inputType": "decimal"}),
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] is None and isinstance(data["error"], str)