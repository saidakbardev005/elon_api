from flask import Flask, request, jsonify
from services.predict_service import (
    get_coordinates,
    predict_price,
    find_best_drivers,
    load_csv_data
)
from transliteration.latin_to_cyrillic import latin_to_cyrillic
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    # 1) Parametrlarni oling
    data   = request.get_json() if request.method == "POST" else request.args
    full_from = data.get("from", "")
    full_to   = data.get("to", "")
    weight    = float(data.get("weight", 0))
    volume    = float(data.get("volume", 0))

    # 2) Foydalanuvchi kiritgan "region,district" -> faqat region qismiga kesib olamiz
    #    misol: "Toshkent,Chorsu" -> "Toshkent"
    frm_region = full_from.split(",")[0].strip()
    to_region  = full_to.split(",")[0].strip()

    # 3) Koordinatalarni olishda ham region qismidan foydalanamiz
    lat, lon = get_coordinates(frm_region)

    # 4) Narx jadvalidagi ustun nomlari va regionlar ro'yxatini tayyorlaymiz
    df_price, _, _, _ = load_csv_data()
    # "From" va "To" ustunlaridagi viloyat nomlarini kichik harflarga o'tkazamiz
    vil_from = df_price["From"].astype(str).str.lower().tolist()
    vil_to   = df_price["To"].astype(str).str.lower().tolist()

    # 5) LabelEncoder’ni faqat viloyat ro‘yxatiga fit qilamiz
    le = LabelEncoder().fit(vil_from + vil_to)

    # 6) Foydalanuvchi kiritgan region nomlarini transliteratsiya + normalize
    frm_cyr   = latin_to_cyrillic(frm_region).strip().lower()
    to_cyr    = latin_to_cyrillic(to_region).strip().lower()

    # 7) Transform; agar region nomi ro'yxatda bo'lmasa xato qaytaring
    try:
        f_enc = int(le.transform([frm_cyr])[0])
        t_enc = int(le.transform([to_cyr])[0])
    except ValueError:
        return jsonify({
            "error": f"Unknown region: from={frm_region}, to={to_region}"
        }), 400

    # 8) Bashorat va haydovchilarni topish
    price   = predict_price(f_enc, t_enc)
    drivers = find_best_drivers(lat, lon, weight, volume)

    # 9) JSON javob qaytaring
    return jsonify({
        "price": price,
        "drivers": drivers
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
