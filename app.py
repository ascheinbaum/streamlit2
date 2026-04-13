import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic - Análisis de Supervivencia", layout="wide")

# ── Carga de datos ──────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_titanic.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["age_class"] = pd.cut(
        df["Age"],
        bins=[0, 2, 13, 18, 30, 60, 120],
        labels=["Infante", "Niño", "Adolescente", "Joven", "Adulto", "Adulto mayor"],
        right=False,
    )
    return df

df = cargar_datos()

st.title("Análisis de Supervivencia — Titanic")

# ── Sidebar: filtros globales ───────────────────────────────────────────────────
st.sidebar.header("Filtros")

sexo = st.sidebar.multiselect("Sexo", options=df["Sex"].unique(), default=list(df["Sex"].unique()))
clase = st.sidebar.multiselect("Clase", options=sorted(df["Pclass"].unique()), default=list(df["Pclass"].unique()))
edad_min, edad_max = st.sidebar.slider(
    "Rango de edad", float(df["Age"].min()), float(df["Age"].max()),
    (float(df["Age"].min()), float(df["Age"].max()))
)
embarked = st.sidebar.multiselect("Puerto de embarque", options=df["Embarked"].dropna().unique(), default=list(df["Embarked"].dropna().unique()))

mask = (
    df["Sex"].isin(sexo) &
    df["Pclass"].isin(clase) &
    df["Age"].between(edad_min, edad_max) &
    df["Embarked"].isin(embarked)
)
dff = df[mask]

st.caption(f"Mostrando **{len(dff)}** pasajeros de {len(df)} totales")

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Resumen descriptivo", "Distribuciones", "Scatter interactivo", "Predictores"])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — Resumen descriptivo
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Métricas generales")

    sobrevivieron = dff[dff["Survived"] == 1]
    no_sobrevivieron = dff[dff["Survived"] == 0]
    tasa = dff["Survived"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pasajeros", len(dff))
    col2.metric("Sobrevivieron", len(sobrevivieron))
    col3.metric("No sobrevivieron", len(no_sobrevivieron))
    col4.metric("Tasa de supervivencia", f"{tasa:.1f}%")

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Supervivencia por sexo**")
        tabla_sexo = dff.groupby("Sex")["Survived"].agg(["sum", "count"])
        tabla_sexo.columns = ["Sobrevivieron", "Total"]
        tabla_sexo["Tasa (%)"] = (tabla_sexo["Sobrevivieron"] / tabla_sexo["Total"] * 100).round(1)
        st.dataframe(tabla_sexo, use_container_width=True)

    with col_b:
        st.markdown("**Supervivencia por clase**")
        tabla_clase = dff.groupby("Pclass")["Survived"].agg(["sum", "count"])
        tabla_clase.columns = ["Sobrevivieron", "Total"]
        tabla_clase["Tasa (%)"] = (tabla_clase["Sobrevivieron"] / tabla_clase["Total"] * 100).round(1)
        st.dataframe(tabla_clase, use_container_width=True)

    with col_c:
        st.markdown("**Supervivencia por rango etario**")
        tabla_age = dff.groupby("age_class", observed=True)["Survived"].agg(["sum", "count"])
        tabla_age.columns = ["Sobrevivieron", "Total"]
        tabla_age["Tasa (%)"] = (tabla_age["Sobrevivieron"] / tabla_age["Total"] * 100).round(1)
        st.dataframe(tabla_age, use_container_width=True)

    st.divider()
    st.subheader("Estadísticas descriptivas por supervivencia")
    vars_num = ["Age", "Fare", "SibSp", "Parch"]
    desc = dff.groupby("Survived")[vars_num].describe().T
    desc.index.names = ["Variable", "Estadístico"]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — Distribuciones
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    col_left, col_right = st.columns([1, 3])

    with col_left:
        tipo = st.radio("Tipo de gráfico", ["Histograma", "Boxplot", "Barras (tasa)", "Violín"])
        variable = st.selectbox("Variable numérica", ["Age", "Fare", "SibSp", "Parch"])
        variable_cat = st.selectbox("Agrupar por", ["Survived", "Sex", "Pclass", "Embarked"])
        bins = st.slider("Bins (histograma)", 5, 60, 20)

    with col_right:
        fig, ax = plt.subplots(figsize=(8, 4))
        colores = {0: "#e74c3c", 1: "#2ecc71"}

        if tipo == "Histograma":
            for val in sorted(dff[variable_cat].dropna().unique()):
                subset = dff[dff[variable_cat] == val][variable].dropna()
                color = colores.get(val, None)
                ax.hist(subset, bins=bins, alpha=0.6, label=str(val), color=color)
            ax.set_xlabel(variable)
            ax.set_ylabel("Frecuencia")
            ax.legend(title=variable_cat)

        elif tipo == "Boxplot":
            data_bp = [dff[dff[variable_cat] == v][variable].dropna() for v in sorted(dff[variable_cat].dropna().unique())]
            labels_bp = [str(v) for v in sorted(dff[variable_cat].dropna().unique())]
            ax.boxplot(data_bp, labels=labels_bp, patch_artist=True,
                       boxprops=dict(facecolor="#3498db", alpha=0.6))
            ax.set_xlabel(variable_cat)
            ax.set_ylabel(variable)

        elif tipo == "Barras (tasa)":
            tasa_var = dff.groupby(variable_cat)["Survived"].mean() * 100
            bars = ax.bar(tasa_var.index.astype(str), tasa_var.values, color="#3498db", edgecolor="white")
            ax.bar_label(bars, fmt="%.1f%%", padding=3)
            ax.set_ylabel("Tasa de supervivencia (%)")
            ax.set_xlabel(variable_cat)
            ax.set_ylim(0, 100)

        elif tipo == "Violín":
            grupos = sorted(dff[variable_cat].dropna().unique())
            data_vio = [dff[dff[variable_cat] == v][variable].dropna().values for v in grupos]
            parts = ax.violinplot(data_vio, positions=range(len(grupos)), showmedians=True)
            ax.set_xticks(range(len(grupos)))
            ax.set_xticklabels([str(g) for g in grupos])
            ax.set_xlabel(variable_cat)
            ax.set_ylabel(variable)

        ax.set_title(f"{variable} por {variable_cat}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — Scatter interactivo
# ────────────────────────────────────────────────────────────────────────────────
with tab3:
    vars_num_scatter = ["Age", "Fare", "SibSp", "Parch", "PassengerId"]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    eje_x = col_s1.selectbox("Eje X", vars_num_scatter, index=0)
    eje_y = col_s2.selectbox("Eje Y", vars_num_scatter, index=1)
    color_por = col_s3.selectbox("Color", ["Survived", "Sex", "Pclass", "Embarked"])
    tamaño_por = col_s4.selectbox("Tamaño", ["Uniforme", "Fare", "Age"])

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    grupos_color = sorted(dff[color_por].dropna().unique())
    palette = sns.color_palette("Set1", len(grupos_color))

    for i, val in enumerate(grupos_color):
        sub = dff[dff[color_por] == val]
        if tamaño_por == "Uniforme":
            sizes = 30
        else:
            sizes = (sub[tamaño_por].fillna(sub[tamaño_por].median()) / sub[tamaño_por].max() * 150 + 10)
        ax2.scatter(sub[eje_x], sub[eje_y], label=str(val), alpha=0.6,
                    color=palette[i], s=sizes, edgecolors="none")

    ax2.set_xlabel(eje_x)
    ax2.set_ylabel(eje_y)
    ax2.set_title(f"{eje_x} vs {eje_y} — coloreado por {color_por}")
    ax2.legend(title=color_por, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — Importancia de variables para predecir supervivencia
# ────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("¿Qué variables predicen mejor la supervivencia?")

    st.markdown(
        "Se entrena un **Random Forest** sobre el dataset filtrado para estimar "
        "la importancia de cada variable. A mayor barra, más útil es la variable para predecir."
    )

    df_model = dff[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].dropna().copy()
    le = LabelEncoder()
    df_model["Sex"] = le.fit_transform(df_model["Sex"])
    df_model["Embarked"] = le.fit_transform(df_model["Embarked"])

    X = df_model.drop("Survived", axis=1)
    y = df_model["Survived"]

    if len(y.unique()) < 2 or len(X) < 10:
        st.warning("No hay suficientes datos con los filtros actuales para entrenar el modelo.")
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)

        importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        colores_imp = ["#e74c3c" if v == importancias.max() else "#3498db" for v in importancias.values]
        bars3 = ax3.barh(importancias.index, importancias.values, color=colores_imp)
        ax3.bar_label(bars3, fmt="%.3f", padding=3)
        ax3.set_xlabel("Importancia (Gini)")
        ax3.set_title("Importancia de variables — Random Forest")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.divider()
        st.subheader("Correlación con supervivencia")
        df_corr = df_model.copy()
        corr = df_corr.corr()["Survived"].drop("Survived").sort_values()

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        colores_corr = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr.values]
        bars4 = ax4.barh(corr.index, corr.values, color=colores_corr)
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.bar_label(bars4, fmt="%.3f", padding=3)
        ax4.set_xlabel("Correlación de Pearson")
        ax4.set_title("Correlación de cada variable con Survived")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
