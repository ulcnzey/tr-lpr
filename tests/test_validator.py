"""Türkiye plaka doğrulama birim testleri."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.validator import (
    ValidationRejectReason,
    PlateKind,
    normalize_ocr_text,
    validate_plate,
)


class TestNormalize:
    def test_strips_spaces_and_separators(self):
        assert normalize_ocr_text("34 ABC 123") == "34ABC123"
        assert normalize_ocr_text("06-AB-1234") == "06AB1234"
        assert normalize_ocr_text("  23 A 12345  ") == "23A12345"

    def test_uppercases(self):
        assert normalize_ocr_text("34abc123") == "34ABC123"

    def test_empty(self):
        assert normalize_ocr_text("") == ""
        assert normalize_ocr_text(None) == ""


class TestValidStandardPlates:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("34 ABC 123", "34ABC123"),  # 3+3
            ("06 AB 1234", "06AB1234"),  # 2+4
            ("23 A 12345", "23A12345"),  # 1+5
            ("01 A 1234", "01A1234"),  # 1+4
            ("34 AB 123", "34AB123"),  # 2+3
            ("34 ABC 12", "34ABC12"),  # 3+2
            ("81 YZ 999", "81YZ999"),
            ("01AA1234", "01AA1234"),
        ],
    )
    def test_valid_formats(self, raw, expected):
        result = validate_plate(raw)
        assert result.is_valid is True
        assert result.plate_text == expected
        assert result.kind == PlateKind.STANDARD
        assert result.province_code == expected[:2]


class TestInvalidProvince:
    @pytest.mark.parametrize("raw", ["00ABC123", "82AB1234", "99A1234", "AB12345"])
    def test_rejects_out_of_range(self, raw):
        result = validate_plate(raw)
        assert result.is_valid is False
        assert result.reject_reason == ValidationRejectReason.INVALID_PROVINCE


class TestInvalidPatterns:
    def test_two_letters_two_digits_not_standard(self):
        # Resmi altı düzende 2 harf + 2 rakam yok
        result = validate_plate("01 AA 12")
        assert result.is_valid is False
        assert result.reject_reason in {
            ValidationRejectReason.INVALID_PATTERN,
            ValidationRejectReason.INVALID_DIGITS,
            ValidationRejectReason.SPECIAL_NOT_SUPPORTED,
        }

    def test_four_letters(self):
        result = validate_plate("34ABCD12")
        assert result.is_valid is False

    def test_forbidden_letter_q(self):
        result = validate_plate("34QA1234")
        assert result.is_valid is False

    def test_forbidden_letter_w(self):
        result = validate_plate("06WX1234")
        assert result.is_valid is False


class TestNoiseAndExtraText:
    @pytest.mark.parametrize(
        "raw",
        [
            "80ILKERCANBAHCECIILKERCANBAHCECIEN922HOODLKAGDLAKALIK",
            "34ABC123OTOMOTIV",
            "GALERI34AB1234",
            "34 ABC 123 OTO SANAYI",
            "HOODLKAGDLAKALIK",
        ],
    )
    def test_rejects_dealer_and_brand_noise(self, raw):
        result = validate_plate(raw)
        assert result.is_valid is False
        assert result.reject_reason in {
            ValidationRejectReason.EXTRA_TEXT,
            ValidationRejectReason.TOO_LONG,
            ValidationRejectReason.INVALID_PATTERN,
            ValidationRejectReason.INVALID_PROVINCE,
        }

    def test_does_not_merge_independent_words_into_plate(self):
        result = validate_plate("OTO 34 ABC 123 GALERI")
        assert result.is_valid is False


class TestEmptyAndShort:
    def test_empty(self):
        result = validate_plate("")
        assert result.is_valid is False
        assert result.reject_reason == ValidationRejectReason.EMPTY

    def test_too_short(self):
        result = validate_plate("34A1")
        assert result.is_valid is False
        assert result.reject_reason == ValidationRejectReason.TOO_SHORT


class TestControlledConfusions:
    def test_o_as_zero_in_digit_region(self):
        # 34 AB 123O → rakam bölgesinde O→0
        result = validate_plate("34AB123O")
        # 2 harf + 4 rakam: AB + 1230
        assert result.is_valid is True
        assert result.plate_text == "34AB1230"
        assert any("O->0" in c for c in result.corrections_applied)

    def test_zero_as_o_in_letter_region(self):
        # 34 0B 1234 → harf bölgesinde 0→O
        result = validate_plate("340B1234")
        assert result.is_valid is True
        assert result.plate_text == "34OB1234"

    def test_does_not_invent_missing_characters(self):
        result = validate_plate("34AB12")  # 2+2 — eksik rakam uydurulmaz
        assert result.is_valid is False

    def test_does_not_blindly_swap_all_similar_chars(self):
        # Harf bölgesinde S bırakılmalı; rastgele 5 yapılmamalı
        result = validate_plate("34SB1234")
        assert result.is_valid is True
        assert result.plate_text == "34SB1234"
        assert result.letter_group == "SB"


class TestRawPreserved:
    def test_raw_and_normalized_kept(self):
        result = validate_plate("34 abc 123")
        assert result.raw_text == "34 abc 123"
        assert result.normalized_text == "34ABC123"
        assert result.is_valid is True


class TestLeadingPrefixTrim:
    def test_strip_one_leading_digit(self):
        # 1-34 BHR 721 -> 134BHR721 -> 34BHR721
        result = validate_plate("1-34 BHR 721")
        assert result.is_valid is True
        assert result.plate_text == "34BHR721"
        assert any(c.startswith("strip_leading:") for c in result.corrections_applied)
        assert "strip_leading:1" in result.corrections_applied

    def test_strip_one_from_compact(self):
        result = validate_plate("134BHR721")
        assert result.is_valid is True
        assert result.plate_text == "34BHR721"
        assert "strip_leading:1" in result.corrections_applied

    def test_strip_two_leading_chars(self):
        result = validate_plate("XX34ABC123")
        assert result.is_valid is True
        assert result.plate_text == "34ABC123"
        assert "strip_leading:XX" in result.corrections_applied

    def test_strip_three_leading_chars(self):
        result = validate_plate("ABC06AB1234")
        assert result.is_valid is True
        assert result.plate_text == "06AB1234"
        assert "strip_leading:ABC" in result.corrections_applied

    def test_does_not_strip_when_already_valid(self):
        result = validate_plate("34ABC123")
        assert result.is_valid is True
        assert result.plate_text == "34ABC123"
        assert not any(c.startswith("strip_leading:") for c in result.corrections_applied)

    def test_does_not_force_invalid_short_body(self):
        result = validate_plate("934AB12")  # strip 1 -> 34AB12 (2+2, standart değil)
        assert result.is_valid is False
