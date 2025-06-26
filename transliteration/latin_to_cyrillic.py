# transliteration/latin_to_cyrillic.py

def latin_to_cyrillic(text: str) -> str:
    """
    Lotincha matnni kirilga oʻgirish:
    1) Avval istisnolar lugʻatida eʼtibor qilamiz.
    2) Aks holda harflarga asoslangan transliteratsiya (qoʻshma birliklar birinchi).
    Matn bosh harfi katta, qolganlari kichik harf bo'ladi.
    """
    # 1) Strip va normalize
    s = text.strip()
    if not s:
        return s

    # 2) Istisnolar: tezkor lugʻat
    # kalitlar hammasi kichik harf bilan berilgan
    EXCEPTIONS = {
        "fargʻona":  "Фарғона",
        "farg'ona":  "Фарғона",
        "fargona":   "Фарғона",
        "toshkent":  "Тошкент",
        "samarkand": "Самарқанд",
        "samarqand": "Самарқанд",
        "shahrisabz":"Шаҳрисабз",
        "qarshi":    "Қарши",
        "namangan":  "Наманган",
        "andijon":   "Андижон",
    }
    key = s.lower()
    if key in EXCEPTIONS:
        return EXCEPTIONS[key]

    # 3) Harf-bosqichma-harf transliteratsiya xaritasi
    mapping = {
        # Qo‘shma birliklar (uzunroq birinchi)
        "yo": "ё", "yu": "ю", "ya": "я", "ye": "е",
        "o‘": "ў", "g‘": "ғ", "o'": "ў", "g'": "ғ",
        "sh": "ш", "ch": "ч", "ng": "нг",
        # Bitta harf
        "a": "а", "b": "б", "d": "д", "e": "э", "f": "ф",
        "g": "г", "h": "ҳ", "i": "и", "j": "ж", "k": "к",
        "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
        "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
        "v": "в", "x": "х", "y": "й", "z": "з",
        # Turli apostroflar
        "’": "", "'": "", "ʻ": "", "`": ""
    }
    # Katta harflar uchun ham qoʻshamiz
    for lat, cyr in list(mapping.items()):
        mapping[lat.upper()]     = cyr.upper()
        mapping[lat.capitalize()] = cyr.capitalize()

    # 4) Modelga ko‘ra transliteratsiya: uzun kalit birinchi
    for latin in sorted(mapping.keys(), key=lambda x: -len(x)):
        s = s.replace(latin, mapping[latin])

    # 5) Natija bosh harf katta, qolgan kichik bo‘lsin
    return s[0].upper() + s[1:].lower() if s else s
