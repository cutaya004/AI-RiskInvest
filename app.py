import streamlit as st 
import numpy as np
import joblib
import matplotlib.pyplot as plt
# ===================== LOAD MODEL =====================
model = joblib.load("riskinvest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== TITLE =====================
st.title("📈 AI-RiskInvest")
st.write("Application de prédiction boursière et gestion du risque")

# ===================== INPUT PRICES =====================
st.subheader("📥 Entrer les 60 derniers prix de clôture")

texte_prix = st.text_area(
    "Entrez les 60 prix (séparés par des virgules ou retour à la ligne)",
    height=200,
    placeholder="Exemple :\n1.25\n1.30\n1.28\n...\n(60 valeurs)"
)

# Liste fixe de 60 prix (دائما باينة)
prices = [0.0] * 60

if texte_prix:
    try:
        texte_prix = texte_prix.replace("\n", ",")
        valeurs = [float(p.strip()) for p in texte_prix.split(",") if p.strip() != ""]

        for i in range(min(len(valeurs), 60)):
            prices[i] = valeurs[i]

        if len(valeurs) != 60:
            st.warning(f"⚠️ Vous avez entré {len(valeurs)} prix. Il faut exactement 60.")
        else:
            st.success("✅ 60 prix chargés avec succès")

    except ValueError:
        st.error("❌ Veuillez entrer uniquement des nombres.")

# ===================== DISPLAY 60 PRICES =====================
st.markdown("### 📋 Détail des 60 prix")

index = 0
for ligne in range(6):
    cols = st.columns(10)
    for col in cols:
        col.number_input(
            f"{index + 1}",
            value=prices[index],
            disabled=True
        )
        index += 1

# ===================== PREDICTION =====================
st.markdown("## 📊 Résultat de la prédiction")

if st.button("Prédire"):
    if len(prices) != 60:
        st.error("❌ Il faut exactement 60 prix pour prédire.")
    else:
        prices_array = np.array(prices).reshape(-1, 1)
        prices_scaled = scaler.transform(prices_array)
        X_input = prices_scaled.reshape(1, -1)

        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(
            prediction.reshape(-1, 1)
        )[0][0]

        st.success("✅ Prédiction effectuée avec succès")
        st.metric("📈 Prix prédit", f"{predicted_price:.4f}")

        # ===================== GRAPH =====================
        st.subheader("📉 Évolution des prix")

        x_prices = list(range(1, 61))
        x_pred = 61

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(x_prices, prices, label="Prix historiques", linewidth=2)
        ax.scatter(x_pred, predicted_price, color="red", label="Prix prédit", zorder=5)
        ax.plot(
            [60, x_pred],
            [prices[-1], predicted_price],
            linestyle="--",
            color="red"
        )

        ax.set_xlabel("Temps")
        ax.set_ylabel("Prix")
        ax.set_title("Prédiction du prochain prix")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

# ===================== CHATBOT =====================
st.divider()
st.subheader("💬 Chatbot AI-RiskInvest")

# ---------- Initialisation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Questions suggérées (Buttons) ----------
st.markdown("### 💡 Questions suggérées")

b1, b2, b3 = st.columns(3)
if b1.button("👋 Hello / Who are you"):
    st.session_state.messages.append({"role": "user", "content": "Hello, who are you?"})
if b2.button("📊 Résultat / Prediction"):
    st.session_state.messages.append({"role": "user", "content": "Explique le résultat de la prédiction"})
if b3.button("⚠️ Risk Management"):
    st.session_state.messages.append({"role": "user", "content": "Quel est le risque de cette prédiction ?"})

b4, b5, b6 = st.columns(3)
if b4.button("📉 RSI / MACD"):
    st.session_state.messages.append({"role": "user", "content": "Utilisez-vous RSI ou MACD ?"})
if b5.button("📰 News du marché"):
    st.session_state.messages.append({"role": "user", "content": "Les news du marché sont-elles prises en compte ?"})
if b6.button("ℹ️ Utilisation de l’app"):
    st.session_state.messages.append({"role": "user", "content": "Comment utiliser l'application ?"})

b7, b8, b9 = st.columns(3)
if b7.button("🧠 Limites du modèle"):
    st.session_state.messages.append({"role": "user", "content": "Quelles sont les limites du modèle ?"})
if b8.button("📚 Données utilisées"):
    st.session_state.messages.append({"role": "user", "content": "Quelles données sont utilisées ?"})
if b9.button("🎓 Questions académiques"):
    st.session_state.messages.append({"role": "user", "content": "Quels sont les objectifs du projet ?"})

# ---------- Affichage historique ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Input utilisateur ----------
user_input = st.chat_input("Posez votre question (FR / EN / AR)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    question = user_input.lower()

    # ---------- LOGIQUE DU CHATBOT ----------

    # HELLO / WHO ARE YOU
    if any(w in question for w in [
        "hello","hi","bonjour","salut","salam",
        "who are you","qui es-tu","من انت","شكون"
    ]):
        reply = (
            "👋 Bonjour / Hello!\n\n"
            "Je suis **AI-RiskInvest** 🤖.\n"
            "J’aide à comprendre les **prédictions**, les **risques**, "
            "les **limites du modèle** et **l’utilisation de l’application**.\n\n"
            "أستطيع المساعدة بالعربية، الفرنسية والإنجليزية."
        )

    # RESULT / PREDICTION
    elif any(w in question for w in [
        "résultat","prediction","prédit","prix",
        "result","predict",
        "نتيجة","توقع","السعر"
    ]):
        if "predicted_price" in locals():
            reply = (
                f"📊 **Résultat de la prédiction**\n\n"
                f"Le prix prédit est **{predicted_price:.4f}**.\n\n"
                "Basé sur les **60 derniers prix historiques**.\n"
                "Il s’agit d’une **estimation** (pas une garantie).\n\n"
                "⚠️ Utilisez toujours une analyse complémentaire."
            )
        else:
            reply = (
                "ℹ️ Aucun résultat disponible.\n"
                "Veuillez d’abord entrer 60 prix et cliquer sur **Prédire**."
            )

    # GOOD / BAD / RELIABILITY
    elif any(w in question for w in [
        "bonne","opportunité","fiable","reliable",
        "good","trust",
        "مزيانة","جيدة","موثوق"
    ]):
        reply = (
            "🧠 **Qualité du résultat**\n\n"
            "La prédiction peut être **utile à court terme** si la tendance est claire.\n"
            "Cependant, elle **n’est pas fiable à 100%**.\n\n"
            "👉 À combiner avec gestion du risque et indicateurs."
        )

    # RISK MANAGEMENT
    elif any(w in question for w in [
        "risque","risk","conseil",
        "خطر","مخاطر","نصح"
    ]):
        reply = (
            "⚠️ **Risk Management**\n\n"
            "• Le marché est imprévisible\n"
            "• Ce n’est **PAS** un conseil financier\n"
            "• Le modèle peut se tromper\n\n"
            "👉 Utilisez toujours : Stop-loss, taille de position, diversification."
        )

    # RSI / MACD / INDICATORS
    elif any(w in question for w in [
        "rsi","macd","indicateur","indicator",
        "مؤشر","مؤشرات"
    ]):
        reply = (
            "📉 **Indicateurs techniques**\n\n"
            "❌ RSI : non utilisé\n"
            "❌ MACD : non utilisé\n\n"
            "Le modèle utilise **uniquement les prix historiques**.\n"
            "Ajouter RSI/MACD améliorerait la précision."
        )

    # DATA USED
    elif any(w in question for w in [
        "données","data","60","normalisé","normalized",
        "بيانات","معطيات"
    ]):
        reply = (
            "📚 **Données utilisées**\n\n"
            "• 60 derniers prix de clôture\n"
            "• Données **normalisées** avant prédiction\n"
            "• Ordre chronologique respecté\n\n"
            "Pourquoi 60 ? Pour capturer la tendance récente."
        )

    # NEWS / MARKET
    elif any(w in question for w in [
        "news","actualité","marché",
        "أخبار","السوق"
    ]):
        reply = (
            "📰 **News du marché**\n\n"
            "Les actualités **ne sont pas intégrées** actuellement.\n"
            "Or, elles peuvent fortement influencer les prix.\n\n"
            "➡️ Une amélioration future peut intégrer l’analyse des news."
        )

    # HOW TO USE
    elif any(w in question for w in [
        "comment","utiliser","how","use",
        "كيف","استعمال"
    ]):
        reply = (
            "ℹ️ **Utilisation de l’application**\n\n"
            "1️⃣ Entrer 60 prix de clôture\n"
            "2️⃣ Vérifier l’affichage\n"
            "3️⃣ Cliquer sur **Prédire**\n"
            "4️⃣ Analyser le prix et le graphique"
        )

    # ACADEMIC QUESTIONS
    elif any(w in question for w in [
        "objectif","goals","choisi","improve","améliorer",
        "أهداف","تطوير","اخترتم"
    ]):
        reply = (
            "🎓 **Questions académiques**\n\n"
            "• Objectif : aider à la décision via le ML\n"
            "• Choix du modèle : simplicité et interprétabilité\n"
            "• Améliorations : RSI, MACD, news, deep learning"
        )

    # DEFAULT
    else:
        reply = (
            "🤖 Je n’ai pas bien compris.\n\n"
            "Exemples :\n"
            "• Hello\n"
            "• Explique le résultat\n"
            "• Quel est le risque ?\n"
            "• RSI / MACD\n"
            "• News du marché"
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)



#_________________________________________________________________________________________3______________________________________________________________________________________
st.markdown("""
<style>

/* Global background */
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}

/* Titles */
h1, h2, h3 {
    color: #f1f5f9;
    font-weight: 600;
}

/* Subtitles */
h4, h5, h6 {
    color: #cbd5f5;
}

/* Buttons */
.stButton > button {
    background-color: #1e293b;
    color: white;
    border-radius: 6px;
    border: 1px solid #475569;
    padding: 8px 16px;
    font-weight: 500;
}
.stButton > button:hover {
    background-color: #334155;
    border-color: #e11d48;
}

/* Inputs */
input {
    background-color: #020617 !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Chat user */
[data-testid="chat-message-user"] {
    background-color: #1e293b;
    border-radius: 10px;
    padding: 8px;
}

/* Chat assistant */
[data-testid="chat-message-assistant"] {
    background-color: #020617;
    border-radius: 10px;
    padding: 8px;
    border-left: 3px solid #e11d48;
}

</style>
""", unsafe_allow_html=True)
