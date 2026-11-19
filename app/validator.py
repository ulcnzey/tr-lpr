"""Türkiye plaka format doğrulama modülü.

Karayolları Trafik Yönetmeliği Madde 55 — standart sivil plaka düzenleri:
  1 harf + 4 rakam
  1 harf + 5 rakam
  2 harf + 3 rakam
  2 harf + 4 rakam
  3 harf + 2 rakam
  3 harf + 3 rakam

İl kodu: 01–81.
Özel / isim plakaları standart sivil olarak kabul edilmez.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PlateKind(str, Enum):
    STANDARD = "standard"
    SPECIAL = "special"
    UNKNOWN = "unknown"


class ValidationRejectReason(str, Enum):
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    INVALID_PROVINCE = "invalid_province"
    INVALID_LETTERS = "invalid_letters"
    INVALID_DIGITS = "invalid_digits"
    INVALID_PATTERN = "invalid_pattern"
    EXTRA_TEXT = "extra_text"
    SPECIAL_NOT_SUPPORTED = "special_not_supported"
    AMBIGUOUS = "ambiguous"


# Plakalarda kullanılmayan harfler (Q, W, X; Türkçe diacritic'ler)
FORBIDDEN_LETTERS = frozenset("QWXÇĞİÖŞÜ")
ALLOWED_LETTERS = frozenset("ABCDEFGHIJKLMNOPRSTUVYZ")

# OCR konum bazlı kontrollü düzeltmeler (gelişigüzel değil)
LETTER_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}

# Standart sivil: (harf_sayısı, rakam_sayısı)
STANDARD_PATTERNS = frozenset(
    {
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
    }
)

# Firma / bayi / reklam benzeri gürültü ipuçları (normalize edilmiş metinde)
NOISE_TOKENS = frozenset(
    {
        "OTO",
        "OTOMOTIV",
        "OTOMOTİV",
        "GALERI",
        "GALERİ",
        "SERVIS",
        "SERVİS",
        "SANAYI",
        "SANAYİ",
        "BAYI",
        "BAYİ",
        "MOTOR",
        "AUTO",
        "AUTOMOTIVE",
        "RENT",
        "RENTACAR",
        "HOOD",
        "TUNING",
        "YIKAMA",
        "LASTIK",
        "LASTİK",
    }
)

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    plate_text: Optional[str]
    raw_text: str
    normalized_text: str
    kind: PlateKind
    province_code: Optional[str] = None
    letter_group: Optional[str] = None
    digit_group: Optional[str] = None
    reject_reason: Optional[ValidationRejectReason] = None
    corrections_applied: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""


def normalize_ocr_text(text: str) -> str:
    """Boşluk ve ayırıcıları temizler; büyük harfe çevirir. Karakter uydurmaz."""
    if text is None:
        return ""
    upper = text.upper().replace("İ", "I").replace("ı", "I")
    upper = _WHITESPACE.sub(" ", upper).strip()
    # Ayırıcıları kaldırarak alfanümerik çekirdek
    compact = _NON_ALNUM.sub("", upper)
    return compact


def _looks_like_embedded_noise(compact: str) -> bool:
    if len(compact) > 12:
        return True
    for token in NOISE_TOKENS:
        if token in compact and not _could_be_plate_core(compact):
            return True
        # Firma adı plaka ile bitişik: ...OTO... veya uzun metin içinde
        if token in compact and len(compact) > 9:
            return True
    return False


def _could_be_plate_core(compact: str) -> bool:
    """Kabaca il+harf+rakam iskeleti var mı?"""
    if len(compact) < 5 or len(compact) > 9:
        return False
    if not compact[:2].isdigit():
        return False
    return True


def _split_province(compact: str) -> Optional[tuple[str, str]]:
    if len(compact) < 5:
        return None
    province = compact[:2]
    if not province.isdigit():
        return None
    code = int(province)
    if code < 1 or code > 81:
        return None
    return province, compact[2:]


def _apply_positional_corrections(body: str) -> tuple[str, tuple[str, ...]]:
    """Harf bölgesinde rakam→harf, rakam bölgesinde harf→rakam düzeltmesi.

    Geçerli bir (harf, rakam) desenine oturtmak için aday bölünmeleri dener.
    Eksik karakter eklemez; yalnızca mevcut karakterleri konum kuralına göre çevirir.
    """
    corrections: list[str] = []
    best: Optional[tuple[str, str, str, list[str]]] = None  # letters, digits, full_body, corr

    for n_letters, n_digits in STANDARD_PATTERNS:
        if len(body) != n_letters + n_digits:
            continue
        letters_raw = body[:n_letters]
        digits_raw = body[n_letters:]
        local_corr: list[str] = []
        letters_chars: list[str] = []
        digits_chars: list[str] = []

        ok = True
        for ch in letters_raw:
            if ch in ALLOWED_LETTERS:
                letters_chars.append(ch)
            elif ch in DIGIT_TO_LETTER and DIGIT_TO_LETTER[ch] in ALLOWED_LETTERS:
                mapped = DIGIT_TO_LETTER[ch]
                letters_chars.append(mapped)
                local_corr.append(f"{ch}->{mapped}@letter")
            else:
                ok = False
                break
        if not ok:
            continue

        for ch in digits_raw:
            if ch.isdigit():
                digits_chars.append(ch)
            elif ch in LETTER_TO_DIGIT:
                mapped = LETTER_TO_DIGIT[ch]
                digits_chars.append(mapped)
                local_corr.append(f"{ch}->{mapped}@digit")
            else:
                ok = False
                break
        if not ok:
            continue

        letters = "".join(letters_chars)
        digits = "".join(digits_chars)
        if any(c in FORBIDDEN_LETTERS for c in letters):
            continue
        if not letters.isalpha() or not digits.isdigit():
            continue

        candidate = (letters, digits, letters + digits, local_corr)
        # Düzeltme sayısı az olanı tercih et
        if best is None or len(local_corr) < len(best[3]):
            best = candidate

    if best is None:
        return body, tuple(corrections)
    return best[2], tuple(best[3])


def _parse_letter_digit_groups(body: str) -> Optional[tuple[str, str, PlateKind]]:
    """Gövdeyi harf + rakam gruplarına ayırır."""
    if not body:
        return None

    # Zaten harf sonra rakam mı?
    m = re.fullmatch(r"([A-Z]+)([0-9]+)", body)
    if m:
        letters, digits = m.group(1), m.group(2)
        if (len(letters), len(digits)) in STANDARD_PATTERNS:
            if all(c in ALLOWED_LETTERS for c in letters):
                return letters, digits, PlateKind.STANDARD
            return None
        # Özel / isim plakası ipucu: çok harf
        if len(letters) >= 4 or len(digits) < 2 or len(digits) > 5:
            return letters, digits, PlateKind.SPECIAL
        return None

    # Rakam/harf karışık → konum düzeltmesi dene
    corrected, _ = _apply_positional_corrections(body)
    m2 = re.fullmatch(r"([A-Z]+)([0-9]+)", corrected)
    if m2:
        letters, digits = m2.group(1), m2.group(2)
        if (len(letters), len(digits)) in STANDARD_PATTERNS:
            if all(c in ALLOWED_LETTERS for c in letters):
                return letters, digits, PlateKind.STANDARD
    return None


def _validate_compact(raw: str, normalized: str) -> ValidationResult:
    """Normalize edilmiş kompakt metni doğrudan doğrular (önek kırpma yok)."""
    split = _split_province(normalized)
    if split is None:
        # İl kodu hatalı veya yok — belki O/0 karışıklığı ilk iki karakterde
        if len(normalized) >= 5:
            fixed_prefix = []
            corr: list[str] = []
            for ch in normalized[:2]:
                if ch.isdigit():
                    fixed_prefix.append(ch)
                elif ch in LETTER_TO_DIGIT:
                    fixed_prefix.append(LETTER_TO_DIGIT[ch])
                    corr.append(f"{ch}->{LETTER_TO_DIGIT[ch]}@province")
                else:
                    fixed_prefix = []
                    break
            if fixed_prefix:
                trial = "".join(fixed_prefix) + normalized[2:]
                split = _split_province(trial)
                if split is not None:
                    province, body = split
                    return _validate_body(
                        raw, trial, province, body, tuple(corr)
                    )

        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            reject_reason=ValidationRejectReason.INVALID_PROVINCE,
            message="İl kodu 01–81 aralığında değil veya okunamadı.",
        )

    province, body = split
    return _validate_body(raw, normalized, province, body, ())


def _try_leading_trim(raw: str, normalized: str) -> Optional[ValidationResult]:
    """OCR'nin plaka önüne eklediği 1–3 karakteri kontrollü kırpar.

    Kalan metnin il kodu 01–81 ve gövdesi STANDARD_PATTERNS ise kabul edilir.
    """
    for n in (1, 2, 3):
        if len(normalized) - n < 5:
            continue
        candidate = normalized[n:]
        split = _split_province(candidate)
        if split is None:
            continue
        province, body = split
        # Gövde uzunluğu standart desenlerden birine oturmalı (hızlı eleme)
        if (len(body) < 3) or all(
            len(body) != n_letters + n_digits
            for n_letters, n_digits in STANDARD_PATTERNS
        ):
            # Yine de konum düzeltmeli yol için _validate_body'ye bırak
            pass
        trimmed = normalized[:n]
        corr = (f"strip_leading:{trimmed}",)
        result = _validate_body(raw, candidate, province, body, corr)
        if result.is_valid:
            return result
    return None


def validate_plate(raw_text: str) -> ValidationResult:
    """OCR çıktısını Türkiye plaka standartlarına göre doğrular."""
    raw = raw_text or ""
    normalized = normalize_ocr_text(raw)

    if not normalized:
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            reject_reason=ValidationRejectReason.EMPTY,
            message="OCR çıktısı boş.",
        )

    if _looks_like_embedded_noise(normalized):
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            reject_reason=ValidationRejectReason.EXTRA_TEXT,
            message="Metin plaka dışı yazı veya aşırı uzun; reddedildi.",
        )

    if len(normalized) < 5:
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            reject_reason=ValidationRejectReason.TOO_SHORT,
            message="Metin plaka için çok kısa.",
        )

    # Standart plaka en fazla 8 karakter; OCR öneki için 11'e kadar izin
    if len(normalized) > 11:
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            reject_reason=ValidationRejectReason.TOO_LONG,
            message="Metin plaka için çok uzun; muhtemel birleşik yazı.",
        )

    result = _validate_compact(raw, normalized)
    if result.is_valid:
        return result

    # Doğrudan geçerli değilse: baştaki 1–3 karakteri kırpmayı dene
    if 6 <= len(normalized) <= 11:
        recovered = _try_leading_trim(raw, normalized)
        if recovered is not None and recovered.is_valid:
            return recovered

    return result


def _validate_body(
    raw: str,
    normalized: str,
    province: str,
    body: str,
    prior_corrections: tuple[str, ...],
) -> ValidationResult:
    corrected_body, pos_corr = _apply_positional_corrections(body)
    corrections = prior_corrections + pos_corr

    # Düzeltme sonrası yeniden ayrıştır
    work_body = corrected_body if pos_corr else body
    parsed = _parse_letter_digit_groups(work_body)

    if parsed is None:
        # Düzeltmeli gövdeyi de dene
        if work_body != body:
            parsed = _parse_letter_digit_groups(body)
        if parsed is None:
            return ValidationResult(
                is_valid=False,
                plate_text=None,
                raw_text=raw,
                normalized_text=normalized,
                kind=PlateKind.UNKNOWN,
                province_code=province,
                reject_reason=ValidationRejectReason.INVALID_PATTERN,
                corrections_applied=corrections,
                message="Geçerli sivil plaka harf/rakam düzenine uymuyor.",
            )

    letters, digits, kind = parsed

    if kind == PlateKind.SPECIAL:
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.SPECIAL,
            province_code=province,
            letter_group=letters,
            digit_group=digits,
            reject_reason=ValidationRejectReason.SPECIAL_NOT_SUPPORTED,
            corrections_applied=corrections,
            message="Özel/isim plakası tespit edildi; standart sivil olarak kabul edilmedi.",
        )

    if any(c in FORBIDDEN_LETTERS for c in letters) or any(
        c not in ALLOWED_LETTERS for c in letters
    ):
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            province_code=province,
            reject_reason=ValidationRejectReason.INVALID_LETTERS,
            corrections_applied=corrections,
            message="Harf grubu geçersiz karakter içeriyor.",
        )

    if not digits.isdigit() or (len(letters), len(digits)) not in STANDARD_PATTERNS:
        return ValidationResult(
            is_valid=False,
            plate_text=None,
            raw_text=raw,
            normalized_text=normalized,
            kind=PlateKind.UNKNOWN,
            province_code=province,
            reject_reason=ValidationRejectReason.INVALID_DIGITS,
            corrections_applied=corrections,
            message="Rakam grubu veya harf/rakam uzunluğu geçersiz.",
        )

    plate_text = f"{province}{letters}{digits}"
    return ValidationResult(
        is_valid=True,
        plate_text=plate_text,
        raw_text=raw,
        normalized_text=normalized,
        kind=PlateKind.STANDARD,
        province_code=province,
        letter_group=letters,
        digit_group=digits,
        corrections_applied=corrections,
        message="Geçerli standart sivil Türkiye plakası.",
    )
