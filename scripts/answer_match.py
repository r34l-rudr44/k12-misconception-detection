"""Does the student's final answer equal the answer key?

In the training data "final answer correct" and the label NONE (no major misconception) are almost the same thing, so
this check carries a large part of the score. It works on transcriptions written as 'L1 | ...' lines and handles the
number formats found in Indonesian school work:

  * currency and units glued to numbers (Rp30.000, 276 cm^2)
  * LaTeX fractions (nested, mixed numbers), roots and superscripts
  * both readings of '3.625' / '3,625' (thousands separator or decimal mark)
  * rounding to the number of decimals of the key
  * transcriber annotations in [brackets] are ignored

Keys that are not a single number (expressions, multiple-choice letters) are compared as normalised strings.

    match(text, key, tail=2) -> 1 if one of the last `tail` lines contains the key, else 0
"""

import re

NUM = re.compile(r"-?\d+(?:[.,]\d+)*(?:\s*/\s*-?\d+(?:[.,]\d+)*)?")
_U = r"rp|rupiah|cm|mm|dm|km|m|kg|gram|gr|g|ton|kuintal|liter|ml|l|meter|siswa|orang|buah|buku|hari|jam|menit|detik|derajat|tahun|bulan|minggu|kali|satuan|unit|persen|lembar|butir|ekor|kotak|batang|potong|anak"
UNITS = rf"(?i)(?<![a-zA-Z])({_U})(?![a-zA-Z])"
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def _num(tok):
    """One reading of a number token (kept for callers that want a single value)."""
    v = _nums(tok)
    return v[0] if v else None


def _plain(t):
    """All plausible values of an unsigned number without '/'. '3.625' and '3,625' are ambiguous -> both readings."""
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", t):
        return [float(t.replace(".", "").replace(",", "."))]  # 1.500.000,00
    if re.fullmatch(r"\d{1,3}(,\d{3})+\.\d+", t):
        return [float(t.replace(",", ""))]  # 1,500,000.00
    if re.fullmatch(r"\d{1,3}([.,]\d{3}){2,}", t):
        return [float(re.sub(r"[.,]", "", t))]  # 1.500.000
    if re.fullmatch(r"\d{1,3}[.,]\d{3}", t):  # 4.000 / 3,625
        return [float(re.sub(r"[.,]", "", t)), float(t.replace(",", "."))]
    if re.fullmatch(r"\d+[.,]\d+", t):
        return [float(t.replace(",", "."))]
    if re.fullmatch(r"\d+", t):
        return [float(t)]
    return []


def _nums(tok):
    tok = tok.replace(" ", "")
    if "/" in tok:
        a, b = tok.split("/", 1)
        return [x / y for x in _nums(a) for y in _nums(b) if y != 0]
    neg = tok.startswith("-")
    return [(-v if neg else v) for v in _plain(tok.lstrip("-"))]


def _clean(s):
    s = str(s).translate(str.maketrans({"−": "-", "–": "-", "⁄": "/"}))
    s = re.sub(r"\[[^\]]*\]", " ", s)  # transcriber annotations: [crossed out], [circled]
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group().translate(_SUP), s).replace("\\pi", "π")
    s = re.sub(r"\\sqrt(\[[^\]]*\])?\s*\{([^{}]*)\}", r"√\2", s)  # before fractions so \frac{3\sqrt{5}}{5} unnests
    s = s.replace("{,}", ",").replace("\\%", "%")
    s = re.sub(r"(\d)\s*(\\[dt]?frac)", r"\1 \2", s)  # mixed number 2\dfrac{5}{8} -> 2 5/8
    for _ in range(4):  # nested fractions, innermost first
        s2 = _FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\sqrt(\[[^\]]*\])?\s*\{([^{}]*)\}", r"√\2", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(left|right|mathrm|displaystyle|circ|quad|,|;|!| )", " ", s)
    s = s.replace("\\times", "×").replace("\\cdot", "·").replace("\\div", ":")
    s = re.sub(r"(?i)\brp\.?\s*(?=\d)", " ", s)  # Rp30.000 / Rp. 30.000
    s = re.sub(rf"(?i)(?<=\d)\s*({_U})\s*(\^\s*\{{?\s*[23]\s*\}}?)?(?![a-zA-Z])", " ", s)  # 30cm, 276 cm^2
    s = re.sub(r"[\$\{\}\\]", " ", s)
    return s


def key_values(key):
    """-> (list of numeric values, None) for a single-number key, else (None, normalised string)."""
    k = _clean(key)
    core = re.sub(UNITS, " ", k)
    mathy = re.search(r"[√×·^π]|\d\s*[a-zA-Z]|[a-zA-Z]\s*\d|[a-zA-Z]\s*\(|[+*:]|\d\s*-\s*\d", core)
    mixed = re.fullmatch(r"\s*(-?)\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*", re.sub(r"[a-zA-Z=%.]+(?!\d)", " ", core))
    if mixed and int(mixed.group(4)):
        v = int(mixed.group(2)) + int(mixed.group(3)) / int(mixed.group(4))
        return [-v if mixed.group(1) else v], None
    words = re.sub(r"\b[a-zA-Z]\s*=", " ", core)  # "x = 5" -> 5
    toks = NUM.findall(words)
    if len(toks) == 1 and not mathy:
        v = _nums(toks[0])
        if v:
            return v, None
    return None, _norm_str(key)


def _norm_str(s):
    s = _clean(s).lower()
    return re.sub(r"[\s()*×·.,;:!?]", "", s)


def lines_of(text):
    out = []
    for l in str(text).splitlines():
        l = re.sub(r"^\s*L\d+\s*\|\s*", "", l)
        if l.strip():
            out.append(l)
    return out


def _line_values(l):
    """Candidate final values on a line: numbers after the last '=', else all numbers; plus mixed numbers."""
    seg = l.split("=")[-1]
    vals = []
    for part in [seg] if NUM.search(seg) else [l]:
        mx = re.compile(r"(-?)\s*(?<![\d.,/])(\d+)\s+(\d+)\s*/\s*(\d+)")
        for m in mx.finditer(part):
            if int(m.group(4)):
                v = int(m.group(2)) + int(m.group(3)) / int(m.group(4))
                vals.append(-v if m.group(1) else v)
        part = mx.sub(" ", part)
        for t in NUM.findall(part):
            vals += _nums(t)
    return vals


def _eq(x, v, decimals):
    if abs(x - v) <= 1e-6 * max(1, abs(v)):
        return True
    return decimals > 0 and abs(round(x, decimals) - v) <= 1e-9 and abs(x - v) < 0.5 * 10**-decimals  # 7,214 ~ 7,21


def _key_decimals(key):
    m = re.search(r"\d[.,](\d{1,2})(?!\d)", _clean(key))
    return len(m.group(1)) if m else 0


def match(text, key, tail=2):
    """1 if the student's final lines contain the key (numerically equivalent), else 0."""
    ls = lines_of(text)
    if not ls:
        return 0
    v, s = key_values(key)
    last = [_clean(l) for l in ls[-tail:]]
    if v is not None:
        d = _key_decimals(key)
        for l in last:
            if any(_eq(x, kv, d) for x in _line_values(l) for kv in v):
                return 1
        return 0
    if not s:
        return 0
    for l in last:
        ln = _norm_str(l)
        if len(s) >= 3:
            if s in ln:
                return 1
        elif re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", re.sub(r"[()*×·.,;:!?]", " ", _clean(l).lower())):
            return 1
    return 0
